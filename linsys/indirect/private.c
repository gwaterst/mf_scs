// #include <Python.h>
// #include "numpy/arrayobject.h"
#include "private.h"
#include "../common.h"
#include "linAlg.h"
#include <stdlib.h>

#define CG_BEST_TOL 1e-7
#define PRINT_INTERVAL 100

/*y = (RHO_X * I + A'A)x */
static void matVec(Data * d, Priv * p, const pfloat * x, pfloat * y);
static idxint pcg(Data *d, Priv * p, const pfloat *s, pfloat * b, idxint max_its, pfloat tol);
// static void transpose(Data * d, Priv * p);
static void resetTmp(Data * d, Priv * p);
static void accScaleDiag(idxint n, const pfloat *diag, const pfloat * x, pfloat * y);
static void scaleDiag(idxint len, const pfloat *diag, pfloat * x);
static void scaleInvDiag(idxint len, const pfloat *diag, pfloat * x);
static void invDiag(idxint n, const pfloat *diag, pfloat * x);
static void getPreconditioner(Data *d, Priv *p);

static idxint totCgIts;
static timer linsysTimer;
static pfloat totalSolveTime;

#include "cones.h"
#include "../amatrix.h"

/* contains routines common to direct and indirect sparse solvers */

#define MIN_SCALE (1e-3)
#define MAX_SCALE (1e3)

// // Vector to numpy array.
// PyObject* vec_to_nparr(const pfloat *data, idxint* length) {
// 	return PyArray_SimpleNewFromData(1, length,
// 							  		NPY_FLOAT64,
// 							  		(void *)data);
// }

/* Ensure val is between MIN_SCALE and MAX_SCALE. */
static pfloat bound(pfloat val, pfloat min_scale, pfloat max_scale) {
	if (val < min_scale) {
		val = min_scale;
	} else if (val > max_scale) {
		val = max_scale;
	}
	return val;
}

// static void lp_mul(Data * d, pfloat * E, pfloat * D_array) {
// 	D_array = vec_to_nparr(D, &(d->m));
// 	if (d->STOCH) {
// 		idxint steps;
// 		for (steps = 0; steps < d->SAMPLES; ++steps) {
// 			E_array = vec_to_nparr(E, &(d->n));
// 			arglist = Py_BuildValue("(OO)", E_array, D_array);
// 			PyObject_CallObject(d->Amul, arglist);
// 			Py_DECREF(arglist);
// 		}
// 	} else {
// 		arglist = Py_BuildValue("(OOi)", E_array, D_array, d->EQUIL_P);
// 		PyObject_CallObject(d->Amul, arglist);
// 		Py_DECREF(arglist);
// 	}
// }

// Populates s with random +1, -1 entries.
void gen_rand_s(pfloat *s, idxint s_len) {
	idxint i;
	for (i = 0; i < s_len; i++) {
		s[i] = (pfloat) 2*(rand() % 2) - 1;
	}
}

// Randomized row norm squared for AE.
// Written to output (assumed length m and zeroed out).
void rand_rnsAE(Data * d, pfloat *E, pfloat *output) {
	idxint i, j;
	for (i = 0; i < d->SAMPLES; i++) {
		gen_rand_s(d->dag_input, d->n);
		scaleDiag(d->n, E, d->dag_input);
		d->Amul(d->fao_dag);
		for (j = 0; j < d->m; ++j) {
			output[j] += d->dag_output[j]*d->dag_output[j]/((pfloat) d->SAMPLES);
		}
	}
}

// Randomized row norm squared for A^TD.
// Written to output (assumed length n and zeroed out).
void rand_rnsATD(Data * d, pfloat *D, pfloat *output) {
	idxint i, j;
	for (i = 0; i < d->SAMPLES; i++) {
		gen_rand_s(d->dag_output, d->m);
		scaleDiag(d->m, D, d->dag_output);
		d->ATmul(d->fao_dag);
		for (j = 0; j < d->n; ++j) {
			output[j] += d->dag_input[j]*d->dag_input[j]/((pfloat) d->SAMPLES);
		}
	}
}

// Computes D and E in a matrix free way.
void normalizeA(Data * d, Priv * p, Work * w, Cone * k) {
	// Probably shouldn't go here TODO.
	if (d->RAND_SEED) {
		srand( time(NULL) );
	} else {
		srand( 1 );
	}
	// import_array();
	// PyObject * E_array;
	// PyObject * D_array;
	// PyObject * arglist;
	pfloat * D = scs_calloc(d->m, sizeof(pfloat));
	pfloat * E = scs_calloc(d->n, sizeof(pfloat));
	idxint i, j, count, delta, *boundaries;//, c1, c2;
	pfloat wrk;//, *nms, e;
	idxint numBoundaries = getConeBoundaries(k, &boundaries);
	pfloat minRowScale = MIN_SCALE * sqrtf((pfloat) d->n), maxRowScale = MAX_SCALE * sqrtf((pfloat) d->n);
	pfloat minColScale = MIN_SCALE * sqrtf((pfloat) d->m), maxColScale = MAX_SCALE * sqrtf((pfloat) d->m);
#ifdef EXTRAVERBOSE
	timer normalizeTimer;
	scs_printf("normalizing A\n");
	tic(&normalizeTimer);
#endif

	// /* calculate row norms */
	// for (i = 0; i < d->n; ++i) {
	// 	c1 = A->p[i];
	// 	c2 = A->p[i + 1];
	// 	for (j = c1; j < c2; ++j) {
	// 		wrk = A->x[j];
	// 		D[A->i[j]] += wrk * wrk;
	// 	}
	// }
	// for (i = 0; i < d->m; ++i) {
	// 	D[i] = sqrt(D[i]); /* just the norms */
	// }

	// Initialize E and D = 1.0.
	for (i = 0; i < d->m; i++) {
		D[i] = 1.0;
	}
	for (i = 0; i < d->n; i++) {
		E[i] = 1.0;
	}
	idxint steps;
	// alpha = n/m, beta = 1.
	pfloat alpha = ((pfloat) d->n)/((pfloat) d->m);
	pfloat beta = 1.0;
	if (d->VERBOSE) {
		printf("STOCH = %i\n", d->STOCH);
		printf("EQUIL_P = %i\n", d->EQUIL_P);
		printf("EQUIL_GAMMA = %f\n", d->EQUIL_GAMMA);
		printf("alpha=%f\n", alpha);
	}
	for (steps = 0; steps < d->EQUIL_STEPS; ++steps) {
		// One iteration of algorithm.
		// resetTmp(d, p);
		// E_array = vec_to_nparr(E, &(d->n));
		// D_array = vec_to_nparr(D, &(d->m));

		/* Set D = alpha*(SCALE*|A|^2diag(E)^2 + alpha^2*gamma*1)^{-1/2}*/

		// Set D = (|A|diag(E))^-1
		// memset(D, 0, d->m * sizeof(pfloat));
		// arglist = Py_BuildValue("(OOiii)", E_array, D_array, d->EQUIL_P, d->STOCH, d->SAMPLES);
		// PyObject_CallObject(d->Amul, arglist);
		// Py_DECREF(arglist);

		// TODO do randomized equil.
		memset(D, 0, d->m * sizeof(pfloat));
		rand_rnsAE(d, E, D);

		/* mean of norms of rows across each cone  */
		// TODO this is wrong, should be sqrt of sum of all
		// squares in the column block that we divide by n_k.
		count = boundaries[0];
		for (i = 1; i < numBoundaries; ++i) {
			wrk = 0;
			delta = boundaries[i];
			for (j = count; j < count + delta; ++j) {
				wrk += D[j];
			}
			wrk /= delta;
			for (j = count; j < count + delta; ++j) {
				D[j] = wrk;
			}
			count += delta;
		}
		// // D *= SCALE.
		// scaleArray(D, d->SCALE, d->m);
		// D += alpha^2*gamma.
		for (i = 0; i < d->m; ++i) {
			D[i] = sqrt(D[i] + alpha*alpha*d->EQUIL_GAMMA);
			D[i] = bound(D[i], minRowScale, maxRowScale);
		}
		// Set D = (D^-1)^-1.
		invDiag(d->m, D, D);
		// D *= alpha
		scaleArray(D,(pfloat) d->n, alpha);

		// /* scale the rows with D */
		// for (i = 0; i < d->n; ++i) {
		// 	for (j = A->p[i]; j < A->p[i + 1]; ++j) {
		// 		A->x[j] /= D[A->i[j]];
		// 	}
		// }

		// /* calculate and scale by col norms, E */
		// for (i = 0; i < d->n; ++i) {
		// 	c1 =  A->p[i + 1] - A->p[i];
		// 	e = calcNorm(&(A->x[A->p[i]]), c1);
		// 	if (e < MIN_SCALE)
		// 		e = 1;
		// 	else if (e > MAX_SCALE)
		// 		e = MAX_SCALE;
		// 	//scaleArray(&(A->x[A->p[i]]), 1.0 / e, c1);
		// 	E[i] = e;
		// }

		/* Set E = beta*(SCALE*|A^T|^2diag(D)^2 + gamma*beta^2*1)^{-1/2} */

		// /* Set E = (|A^T|diag(D))^-1 */
		// E_array = vec_to_nparr(E, &(d->n));
		// D_array = vec_to_nparr(D, &(d->m));
		// memset(E, 0, d->n * sizeof(pfloat));
		// arglist = Py_BuildValue("(OOiii)", D_array, E_array, d->EQUIL_P, d->STOCH, d->SAMPLES);
		// PyObject_CallObject(d->ATmul, arglist);
		// Py_DECREF(arglist);
		memset(E, 0, d->n * sizeof(pfloat));
		rand_rnsATD(d, D, E);
		// // E *= SCALE.
		// scaleArray(E, d->SCALE, d->n);
		// E += beta^2*gamma.
		for (i = 0; i < d->n; ++i) {
			p->M[i] = sqrt(E[i]);
			E[i] = sqrt(E[i] + beta*beta*d->EQUIL_GAMMA);
			E[i] = bound(E[i], minColScale, maxColScale);
			p->M[i] /= E[i];
		}
		invDiag(d->n, E, E);
		// E *= beta
		scaleArray(E, (pfloat) d->m, beta);
	}
	scs_free(boundaries);

	// // TODO touches A.
	// nms = scs_calloc(d->m, sizeof(pfloat));
	// for (i = 0; i < d->n; ++i) {
	// 	for (j = A->p[i]; j < A->p[i + 1]; ++j) {
	// 		wrk = A->x[j]/(D[A->i[j]]*E[i]);
	// 		nms[A->i[j]] += wrk * wrk;
	// 	}
	// }
	// w->meanNormRowA = 0.0;
	// for (i = 0; i < d->m; ++i) {
	// 	w->meanNormRowA += sqrt(nms[i]) / d->m;
	// }

	// scs_free(nms);
	// TODO touches A.
	// if (d->SCALE != 1) {
	// 	scaleArray(A->x, d->SCALE, A->p[d->n]);
	// }
	// for (i = 0; i < d->m; i++) {
	// 	printf("D[%d]=%f\n", i, D[i]);
	// }
	// for (i = 0; i < d->n; i++) {
	// 	printf("E[%d]=%f\n", i, E[i]);
	// }

	// TODO trying with row norm squared = n
	// and col norm squared = m.

	// meanNormRowA = D ||A||_2 E1/m
	resetTmp(d, p);
	// E_array = vec_to_nparr(E, &(d->n));
	// D_array = vec_to_nparr(p->tmp_m, &(d->m));
	// arglist = Py_BuildValue("(OOiii)", E_array, D_array, d->EQUIL_P, d->STOCH, d->SAMPLES);
	// PyObject_CallObject(d->Amul, arglist);
	// Py_DECREF(arglist);
	rand_rnsAE(d, E, p->tmp_m);
	// Convert to L2 norm from norm squared.
	for (i = 0; i < d->m; ++i) {
		p->tmp_m[i] = sqrt(p->tmp_m[i]);
	}
	// // Scale by SCALE.
	// scaleArray(p->tmp_m, d->SCALE, d->m);
	// Scale by D.
	scaleDiag(d->m, D, p->tmp_m);
	w->meanNormRowA = 0.0;
	for (i = 0; i < d->m; ++i) {
	   	w->meanNormRowA += p->tmp_m[i] / (pfloat) d->m;
	}
	// w->meanNormRowA = alpha;
	if (d->VERBOSE)
		printf("w->meanNormRowA=%f\n", w->meanNormRowA);

	/* calculate mean of col norms of A */
	// meanNormColA = E ||A^T||_2 D1/m
	if (d->EQUIL_STEPS == 0) {
		// resetTmp(d, p);
		// E_array = vec_to_nparr(p->tmp_n, &(d->n));
		// D_array = vec_to_nparr(D, &(d->m));
		// arglist = Py_BuildValue("(OOiii)", D_array, E_array, d->EQUIL_P, d->STOCH, d->SAMPLES);
		// PyObject_CallObject(d->ATmul, arglist);
		// Py_DECREF(arglist);
		rand_rnsATD(d, D, p->tmp_n);
		// // Scale by SCALE.
		// scaleArray(p->tmp_m, d->SCALE, d->m);
		// Scale by D.
		scaleDiag(d->n, E, p->tmp_n);
		w->meanNormColA = 0.0;
		for (i = 0; i < d->n; ++i) {
			// Save this result and reuse for preconditioner.
			p->M[i] = p->tmp_n[i];
			w->meanNormColA += p->tmp_n[i] / (float) d->n;
		}
	}
	w->meanNormColA = 0;
	for (i = 0; i < d->n; ++i) {
		// Save this result and reuse for preconditioner.
		// p->M[i] = beta;
		w->meanNormColA += p->M[i]/(pfloat) d->n;
	}
	// w->meanNormColA = beta;
	if (d->VERBOSE)
		printf("w->meanNormColA=%f\n", w->meanNormColA);

	// Set D = D^-1 and E = E^-1 because assumed inverted elsewhere.
	invDiag(d->m, D, D);
	invDiag(d->n, E, E);

	// Debugging info.
	if (d->VERBOSE) {
		printf("D[0]=%f\n", D[0]);
		printf("E[0]=%f\n", E[0]);
	}
	pfloat avg = 0;
	for (i = 0; i < d->m; ++i) {
		avg += D[i] / d->m;
	}
	if (d->VERBOSE)
		printf("D average=%f\n", avg);
	avg = 0;
	for (i = 0; i < d->n; ++i) {
		avg += E[i] / d->n;
	}
	if (d->VERBOSE)
		printf("E average=%f\n", avg);

	w->D = D;
	w->E = E;
	// Also store in data.
	p->D = D;
	p->E = E;

#ifdef EXTRAVERBOSE
	scs_printf("finished normalizing A, time: %6f s\n", tocq(&normalizeTimer) / 1e3);
#endif
	getPreconditioner(d, p);
}

void unNormalizeA(Data *d, Work * w) {
	// No-op.
}

char * getLinSysMethod(Data * d, Priv * p) {
	char * str = scs_malloc(sizeof(char) * 128);
	sprintf(str, "sparse-indirect, nnz in A = ???, CG tol ~ 1/iter^(%2.2f)", d->CG_RATE);
	return str;
}

char * getLinSysSummary(Priv * p, Info * info) {
	char * str = scs_malloc(sizeof(char) * 128);
	sprintf(str, "\tLin-sys: avg # CG iterations: %2.2f, avg solve time: %1.2es\n",
			(pfloat ) totCgIts / (info->iter + 1), totalSolveTime / (info->iter + 1) / 1e3);
	totCgIts = 0;
	totalSolveTime = 0;
	return str;
}

/* M = inv ( diag ( RHO_X * I + A'A ) ) */
/* M = inv ( diag ( RHO_X * I + (EA'D)(DAE)*SCALE^2 ) ) */
void getPreconditioner(Data *d, Priv *p) {
	// import_array();
	idxint i;
	pfloat * M = p->M;
	// PyObject* E_array;
	// PyObject* D_array;

#ifdef EXTRAVERBOSE
	scs_printf("getting pre-conditioner\n");
#endif

	// resetTmp(d, p);
	// // tmp_m = D^-1
	// invDiag(d->m, p->D, p->tmp_m);
	// D_array = vec_to_nparr(p->tmp_m, &(d->m));
	// E_array = vec_to_nparr(p->tmp_n, &(d->n));
	// PyObject *arglist;
	// arglist = Py_BuildValue("(OOiii)", D_array, E_array, d->EQUIL_P, d->STOCH, d->SAMPLES);
	// PyObject_CallObject(d->ATmul, arglist);
	// Py_DECREF(arglist);
	// // Scale by SCALE.
	// scaleArray(p->tmp_n, d->SCALE, d->n);
	// // Scale by E.
	// scaleInvDiag(d->n, p->E, p->tmp_n);

	pfloat avg = 0;
	for (i = 0; i < d->n; ++i) {
		if (d->PRECOND) {
			// float test = 1 / (d->RHO_X + calcNormSq(&(A->x[A->p[i]]), A->p[i + 1] - A->p[i]));
			// M[i] = 1 / (d->RHO_X + p->tmp_n[i]*p->tmp_n[i]);
			M[i] = 1 / (d->RHO_X + d->SCALE*d->SCALE*p->M[i]*p->M[i]);
			// M[i] = 1 / (d->RHO_X + d->SCALE*d->SCALE*(pfloat) d->m);
		} else {
			M[i] = 1;
		}

		avg += M[i]/d->n;
	}
	// Clean up arrays.
	// Py_DECREF(arglist);

	// Debug.
	if (d->VERBOSE) {
		printf("M[0]=%f\n", M[0]);
		printf("M average = %f\n", avg);
	}

#ifdef EXTRAVERBOSE
	scs_printf("finished getting pre-conditioner\n");
#endif

}

Priv * initPriv(Data * d) {
	AMatrix * A = d->A;
	Priv * p = scs_calloc(1, sizeof(Priv));
	p->p = scs_malloc((d->n) * sizeof(pfloat));
	p->r = scs_malloc((d->n) * sizeof(pfloat));
	p->Ap = scs_malloc((d->n) * sizeof(pfloat));
	p->tmp_n = scs_malloc((d->n) * sizeof(pfloat));
	p->tmp_m = scs_malloc((d->m) * sizeof(pfloat));
	p->tmp_matvec = scs_malloc((d->m) * sizeof(pfloat));

	/* preconditioner memory */
	p->z = scs_malloc((d->n) * sizeof(pfloat));
	p->M = scs_malloc((d->n) * sizeof(pfloat));

	// p->Ati = scs_malloc((A->p[d->n]) * sizeof(idxint));
	// p->Atp = scs_malloc((d->m + 1) * sizeof(idxint));
	// p->Atx = scs_malloc((A->p[d->n]) * sizeof(pfloat));
	// transpose(d, p);
	totalSolveTime = 0;
	totCgIts = 0;
	if (!p->p || !p->r || !p->Ap || !p->tmp_n || !p->tmp_m ||
	    !p->tmp_matvec) {// || !p->Ati || !p->Atp || !p->Atx) {
		freePriv(p);
		return NULL;
	}
	return p;
}

static void transpose(Data * d, Priv * p) {
	idxint * Ci = p->Ati;
	idxint * Cp = p->Atp;
	pfloat * Cx = p->Atx;
	idxint m = d->m;
	idxint n = d->n;

	idxint * Ap = d->A->p;
	idxint * Ai = d->A->i;
	pfloat * Ax = d->A->x;

	idxint i, j, q, *z, c1, c2;
#ifdef EXTRAVERBOSE
	timer transposeTimer;
	scs_printf("transposing A\n");
	tic(&transposeTimer);
#endif

	z = scs_calloc(m, sizeof(idxint));
	for (i = 0; i < Ap[n]; i++)
		z[Ai[i]]++; /* row counts */
	cs_cumsum(Cp, z, m); /* row pointers */

	for (j = 0; j < n; j++) {
		c1 = Ap[j];
		c2 = Ap[j + 1];
		for (i = c1; i < c2; i++) {
			q = z[Ai[i]];
			Ci[q] = j; /* place A(i,j) as entry C(j,i) */
			Cx[q] = Ax[i];
			z[Ai[i]]++;
		}
	}
	scs_free(z);

#ifdef EXTRAVERBOSE
	scs_printf("finished transposing A, time: %6f s\n", tocq(&transposeTimer) / 1e3);
#endif

}

void freePriv(Priv * p) {
	if (p) {
		if (p->p)
			scs_free(p->p);
		if (p->r)
			scs_free(p->r);
		if (p->Ap)
			scs_free(p->Ap);
		if (p->tmp_n)
			scs_free(p->tmp_n);
		if (p->tmp_m)
			scs_free(p->tmp_m);
		if (p->tmp_matvec)
			scs_free(p->tmp_matvec);
		if (p->Ati)
			scs_free(p->Ati);
		if (p->Atx)
			scs_free(p->Atx);
		if (p->Atp)
			scs_free(p->Atp);
		if (p->z)
			scs_free(p->z);
		if (p->M)
			scs_free(p->M);
		scs_free(p);
	}
}

void solveLinSys(Data *d, Priv * p, pfloat * b, const pfloat * s, idxint iter, idxint *cgIters) {
	idxint cgIts;
	pfloat cgTol = calcNorm(b, d->n) * (iter < 0 ? CG_BEST_TOL : 1 / POWF(iter + 1, d->CG_RATE));

#ifdef EXTRAVERBOSE
	scs_printf("solving lin sys\n");
#endif

	cgTol = MAX(cgTol, CG_BEST_TOL);
	tic(&linsysTimer);
	/* solves Mx = b, for x but stores result in b */
	/* s contains warm-start (if available) */
	accumByAtrans(d, p, &(b[d->n]), b);
	/* solves (I+A'A)x = b, s warm start, solution stored in b */
	cgIts = pcg(d, p, s, b, d->n, MAX(cgTol, CG_BEST_TOL));
	scaleArray(&(b[d->n]), -1, d->m);
	if (isnan(b[0])) {
		exit(1);
	}
	accumByA(d, p, b, &(b[d->n]));

#ifdef EXTRAVERBOSE
	scs_printf("\tCG iterations: %i\n", (int) cgIts);
#endif
	if (iter >= 0) {
		*cgIters += cgIts;
		totCgIts += cgIts;
	}

	totalSolveTime += tocq(&linsysTimer);
}

static void applyPreConditioner(pfloat * M, pfloat * z, pfloat * r, idxint n, pfloat *ipzr) {
	idxint i;
	*ipzr = 0;
	for (i = 0; i < n; ++i) {
		z[i] = r[i] * M[i];
		*ipzr += z[i] * r[i];
	}
}

static idxint pcg(Data *d, Priv * pr, const pfloat * s, pfloat * b, idxint max_its, pfloat tol) {
	/* solves (I+A'A)x = b */
	/* warm start cg with s */
	idxint i, n = d->n;
	pfloat ipzr, ipzrOld, alpha;
	pfloat *p = pr->p; /* cg direction */
	pfloat *Ap = pr->Ap; /* updated CG direction */
	pfloat *r = pr->r; /* cg residual */
	pfloat *z = pr->z; /* for preconditioning */
	pfloat *M = pr->M; /* inverse diagonal preconditioner */

	if (s == NULL) {
		memcpy(r, b, n * sizeof(pfloat));
		memset(b, 0, n * sizeof(pfloat));
	} else {
		matVec(d, pr, s, r);
		addScaledArray(r, b, n, -1);
		scaleArray(r, -1, n);
		memcpy(b, s, n * sizeof(pfloat));
	}

	/* check to see if we need to run CG at all */
	if (calcNorm(r, n) < MIN(tol, 1e-18)) {
	    return 0;
	}

	applyPreConditioner(M, z, r, n, &ipzr);
	memcpy(p, z, n * sizeof(pfloat));

	for (i = 0; i < max_its; ++i) {
		// printf("p[0] = %f, Ap[0] = %f\n", p[0], Ap[0]);
		matVec(d, pr, p, Ap);
		// printf("p[0] = %f, Ap[0] = %f\n", p[0], Ap[0]);
		alpha = ipzr / innerProd(p, Ap, n);
		// printf("alpha=%f, innerProd = %f\n", alpha, innerProd(p, Ap, n));
		addScaledArray(b, p, n, alpha);
		addScaledArray(r, Ap, n, -alpha);

		if (calcNorm(r, n) < tol) {
			// scs_printf("tol: %.4e, resid: %.4e, iters: %i\n", tol, calcNorm(r, n), i+1);
			return i + 1;
		}
		ipzrOld = ipzr;
		applyPreConditioner(M, z, r, n, &ipzr);

		scaleArray(p, ipzr / ipzrOld, n);
		addScaledArray(p, z, n, 1);
	}
	return i;
}

/*  y += diag*x */
static void accScaleDiag(idxint len, const pfloat *diag, const pfloat * x, pfloat * y) {
	idxint i;
	for (i = 0; i < len; i++) {
		y[i] += x[i]/diag[i];
	}
}

/*  x = diag*x */
static void scaleDiag(idxint len, const pfloat *diag, pfloat * x) {
	idxint i;
	for (i = 0; i < len; i++) {
		x[i] = diag[i]*x[i];
	}
}

/*  x = diag^-1*x */
static void scaleInvDiag(idxint len, const pfloat *diag, pfloat * x) {
	idxint i;
	for (i = 0; i < len; i++) {
		x[i] = x[i]/diag[i];
	}
}

/*  x = diag^-1*1 */
static void invDiag(idxint len, const pfloat *diag, pfloat * x) {
	idxint i;
	for (i = 0; i < len; i++) {
		x[i] = 1.0/diag[i];
	}
}

/* Resets tmp_n and tmp_m to zeros. */
static void resetTmp(Data * d, Priv * p) {
	memset(p->tmp_m, 0, d->m * sizeof(pfloat));
	memset(p->tmp_n, 0, d->n * sizeof(pfloat));
}

/*y = (RHO_X * I + A'A)x */
static void matVec(Data * d, Priv * p, const pfloat * x, pfloat * y) {
	pfloat * tmp = p->tmp_matvec;
	if isnan(x[0]) {
		return;
	}
	// printf("1. norm x %f\n", calcNorm(x, d->n));
	memset(tmp, 0, d->m * sizeof(pfloat));
	// pfloat pre = calcNorm(tmp, d->m);
	accumByA(d, p, x, tmp);
	// printf("2. norm tmp %f -> %f\n", pre, calcNorm(tmp, d->m));
	memset(y, 0, d->n * sizeof(pfloat));
	accumByAtrans(d, p, tmp, y);
	// printf("3. norm y %f\n", calcNorm(y, d->n));
	addScaledArray(y, x, d->n, d->RHO_X);
	// printf("4. norm y %f\n", calcNorm(y, d->n));
}

void _accumByAtrans(idxint n, pfloat * Ax, idxint * Ai, idxint * Ap, const pfloat *x, pfloat *y) {
	/* y  = A'*x
	 A in column compressed format
	 parallelizes over columns (rows of A')
	 */
	idxint p, j;
	idxint c1, c2;
	pfloat yj;
#ifdef OPENMP
#pragma omp parallel for private(p,c1,c2,yj)
#endif
	for (j = 0; j < n; j++) {
		yj = y[j];
		c1 = Ap[j];
		c2 = Ap[j + 1];
		for (p = c1; p < c2; p++) {
			yj += Ax[p] * x[Ai[p]];
		}
		y[j] = yj;
	}
}

// y += EA'D*SCALE*x
void accumByAtrans(Data * d, Priv * p, const pfloat *x, pfloat *y) {
	// // Create arrays for x, y.
	// pfloat *z = malloc(sizeof(pfloat)*(d->n));
	// memcpy(z, y, sizeof(pfloat)*(d->n));
	// resetTmp(d, p);
	memset(p->tmp_m, 0, d->m * sizeof(pfloat));
	// PyObject* x_array;
	// PyObject* y_array;
	if (d->NORMALIZE) {
		// tmp_m = D*x.
		accScaleDiag(d->m, p->D, x, p->tmp_m);
		// tmp_m *= SCALE.
		scaleArray(p->tmp_m, d->SCALE, d->m);
		// x_array = vec_to_nparr(p->tmp_m, &(d->m));
		// y_array = vec_to_nparr(p->tmp_n, &(d->n));
	} else {
		// tmp_m += SCALE*x.
		addScaledArray(p->tmp_m, x, d->SCALE, d->m);
		// x_array = vec_to_nparr(p->tmp_m, &(d->m));
		// y_array = vec_to_nparr(y, &(d->n));
	}
	// PyObject *arglist;
	// arglist = Py_BuildValue("(OO)", x_array, y_array);
	// PyObject_CallObject(d->ATmul, arglist);
	// Py_DECREF(arglist);
	memcpy(d->dag_output, p->tmp_m, d->m*sizeof(pfloat));
	d->ATmul(d->fao_dag);
	// y += E*tmp_n.
	if (d->NORMALIZE) {
		accScaleDiag(d->n, p->E, d->dag_input, y);
	}

	// AMatrix * A = d->A;
	// resetTmp(d, p);
	// accScaleDiag(d->m, p->D, x, p->tmp_m);
	// _accumByAtrans(d->n, A->x, A->i, A->p, p->tmp_m, p->tmp_n);
	// accScaleDiag(d->n, p->E, p->tmp_n, z);
	// for (int i=0; i < d->n; i++) {
	// 	if (abs(z[i] - y[i]) > 1e-4) {
	// 		scs_printf("x vals %6f, %6f \n", x[0], x[1]);
	// 		scs_printf("z val %6f, y val %6f \n", z[i], y[i]);
	// 		scs_printf("difference %12f at %i\n", z[i] - y[i], i);
	// 	}
	// }
	// free(z);
}

// y += DAE*SCALE*x
void accumByA(Data * d, Priv * p, const pfloat *x, pfloat *y) {
	// // Create arrays for x, y.
	// // pfloat *z = malloc(sizeof(pfloat)*(d->m));
	// // memcpy(z, y, sizeof(pfloat)*(d->m));
	// PyObject* x_array;
	// PyObject* y_array;
	// resetTmp(d, p);
	memset(p->tmp_n, 0, d->n * sizeof(pfloat));
	if (d->NORMALIZE) {
		// tmp_n = E*x.
		accScaleDiag(d->n, p->E, x, p->tmp_n);
		// tmp_n *= SCALE.
		scaleArray(p->tmp_n, d->SCALE, d->n);
		// x_array = vec_to_nparr(p->tmp_n, &(d->n));
		// y_array = vec_to_nparr(p->tmp_m, &(d->m));
	} else {
		// tmp_n += SCALE*x.
		addScaledArray(p->tmp_n, x, d->SCALE, d->n);
		// x_array = vec_to_nparr(p->tmp_n, &(d->n));
		// y_array = vec_to_nparr(y, &(d->m));
	}
	// PyObject *arglist;
	// arglist = Py_BuildValue("(OO)", x_array, y_array);
	// PyObject_CallObject(d->Amul, arglist);
	// Py_DECREF(arglist);
	memcpy(d->dag_input, p->tmp_n, d->n*sizeof(pfloat));
	d->Amul(d->fao_dag);

	// y += D*tmp_m.
	if (d->NORMALIZE) {
		// printf("accumByA dag_output[0] = %f\n", d->dag_output[0]);
		accScaleDiag(d->m, p->D, d->dag_output, y);
	}
	// AMatrix * A = d->A;
	// _accumByAtrans(d->m, p->Atx, p->Ati, p->Atp, x, y);
	// for (int i=0; i < d->m; i++) {
	// 	if (fabs(z[i] - y[i]) > 0.00001) {
	// 		scs_printf("x vals %6f, %6f, %6f, %6f \n", x[0], x[1], x[2], x[3]);
	// 		scs_printf("z val %6f, y val %6f \n", z[i], y[i]);
	// 		scs_printf("difference %12f at %i\n", z[i] - y[i], i);
	// 	}
	// }
	// free(z);
}
