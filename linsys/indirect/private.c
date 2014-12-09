#include <Python.h>
#include "numpy/arrayobject.h"
#include "private.h"
#include "../common.h"
#include "linAlg.h"

#define CG_BEST_TOL 1e-7
#define PRINT_INTERVAL 100

/*y = (RHO_X * I + A'A)x */
static void matVec(Data * d, Priv * p, const pfloat * x, pfloat * y);
static idxint pcg(Data *d, Priv * p, const pfloat *s, pfloat * b, idxint max_its, pfloat tol);
static void transpose(Data * d, Priv * p);

static idxint totCgIts;
static timer linsysTimer;
static pfloat totalSolveTime;

#include "cones.h"
#include "../amatrix.h"

/* contains routines common to direct and indirect sparse solvers */

#define MIN_SCALE 1e-2
#define MAX_SCALE 1e3

// Vector to numpy array.
PyObject* vec_to_nparr(const pfloat *data, idxint* length) {
	return PyArray_SimpleNewFromData(1, length,
							  		NPY_FLOAT64,
							  		(void *)data);
}

// Computes D and E in a matrix free way.
void normalizeA(Data * d, Work * w, Cone * k) {
	import_array();
	AMatrix * A = d->A;
	pfloat * D = scs_calloc(d->m, sizeof(pfloat));
	pfloat * E = scs_calloc(d->n, sizeof(pfloat));
	idxint i, j, count, delta, *boundaries, c1, c2;
	pfloat wrk, *nms, e;
	idxint numBoundaries = getConeBoundaries(k, &boundaries);
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
	/* Get D = |A|1 */
	for (idxint i = 0; i < d->n; ++i) {
		E[i] = 1;
	}
	PyObject* E_array = vec_to_nparr(E, &(d->n));
	PyObject* D_array = vec_to_nparr(D, &(d->m));
	PyObject *arglist;
	arglist = Py_BuildValue("(OOi)", E_array, D_array, Py_True);
	PyObject_CallObject(d->Amul, arglist);
	Py_DECREF(arglist);
	/* mean of norms of rows across each cone  */

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
	scs_free(boundaries);

	for (i = 0; i < d->m; ++i) {
		if (D[i] < MIN_SCALE) {
			D[i] = 1;
		} else if (D[i] > MAX_SCALE) {
			D[i] = MAX_SCALE;
		}
	}
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
	/* Set E = |A^T|diag(D) */
	scaleArray(E, 0, d->n);
	arglist = Py_BuildValue("(OOi)", D_array, E_array, Py_True);
	PyObject_CallObject(d->ATmul, arglist);
	Py_DECREF(arglist);

	// TODO touches A.
	nms = scs_calloc(d->m, sizeof(pfloat));
	for (i = 0; i < d->n; ++i) {
		for (j = A->p[i]; j < A->p[i + 1]; ++j) {
			wrk = A->x[j]/(D[A->i[j]]*E[i]);
			nms[A->i[j]] += wrk * wrk;
		}
	}
	w->meanNormRowA = 0.0;
	for (i = 0; i < d->m; ++i) {
		w->meanNormRowA += sqrt(nms[i]) / d->m;
	}
	scs_free(nms);
	// TODO touches A.
	// if (d->SCALE != 1) {
	// 	scaleArray(A->x, d->SCALE, A->p[d->n]);
	// }

	w->D = D;
	w->E = E;
	// Also store in data.
	d->D = D;
	d->E = E;

#ifdef EXTRAVERBOSE
	scs_printf("finished normalizing A, time: %6f s\n", tocq(&normalizeTimer) / 1e3);
#endif
}

void unNormalizeA(Data *d, Work * w) {
	// No-op.
}

char * getLinSysMethod(Data * d, Priv * p) {
	char * str = scs_malloc(sizeof(char) * 128);
	sprintf(str, "sparse-indirect, nnz in A = %li, CG tol ~ 1/iter^(%2.2f)", (long ) d->A->p[d->n], d->CG_RATE);
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
void getPreconditioner(Data *d, Priv *p) {
	import_array();
	idxint i;
	pfloat * M = p->M;
	AMatrix * A = d->A;

	pfloat * x = malloc(sizeof(pfloat)*(d->n));
	pfloat * y = malloc(sizeof(pfloat)*(d->m));
	PyObject * x_array = vec_to_nparr(x, &(d->n));
	PyObject * y_array = vec_to_nparr(y, &(d->m));
	PyObject *arglist;
	arglist = Py_BuildValue("(OO)", x_array, y_array);

#ifdef EXTRAVERBOSE
	scs_printf("getting pre-conditioner\n");
#endif

	for (i = 0; i < d->n; ++i) {
		// x[i] = 1.0;
		// PyObject_CallObject(d->Amul, arglist);
		// M[i] = 1 / (d->RHO_X + calcNormSq(y, d->m));
		float test = 1 / (d->RHO_X + calcNormSq(&(A->x[A->p[i]]), A->p[i + 1] - A->p[i]));
		M[i] = 1; //TODO test is wrong.
		// if (M[i] - test > 0.0) {
		// 	scs_printf("difference %12f \n", M[i]-test);
		// }
		/* M[i] = 1; */
		// // zero out x and y.
		// scaleArray(x, 0.0, d->n);
		// scaleArray(y, 0.0, d->m);
	}
	// Clean up x, y, etc.
	Py_DECREF(arglist);
	free(x);
	free(y);

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

	p->Ati = scs_malloc((A->p[d->n]) * sizeof(idxint));
	p->Atp = scs_malloc((d->m + 1) * sizeof(idxint));
	p->Atx = scs_malloc((A->p[d->n]) * sizeof(pfloat));
	transpose(d, p);
	getPreconditioner(d, p);
	totalSolveTime = 0;
	totCgIts = 0;
	if (!p->p || !p->r || !p->Ap || !p->tmp_n || !p->tmp_m ||
	    !p->tmp_matvec || !p->Ati || !p->Atp || !p->Atx) {
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

void solveLinSys(Data *d, Priv * p, pfloat * b, const pfloat * s, idxint iter) {
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
	cgIts = pcg(d, p, s, b, d->n, cgTol);
	scaleArray(&(b[d->n]), -1, d->m);
	accumByA(d, p, b, &(b[d->n]));

#ifdef EXTRAVERBOSE
	scs_printf("\tCG iterations: %i\n", (int) cgIts);
#endif
	if (iter >= 0) {
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
		memset(b, 0.0, n * sizeof(pfloat));
	} else {
		matVec(d, pr, s, r);
		addScaledArray(r, b, n, -1);
		scaleArray(r, -1, n);
		memcpy(b, s, n * sizeof(pfloat));
	}
	applyPreConditioner(M, z, r, n, &ipzr);
	memcpy(p, z, n * sizeof(pfloat));

	for (i = 0; i < max_its; ++i) {
		matVec(d, pr, p, Ap);

		alpha = ipzr / innerProd(p, Ap, n);
		addScaledArray(b, p, n, alpha);
		addScaledArray(r, Ap, n, -alpha);

		if (calcNorm(r, n) < tol) {
			/*scs_printf("tol: %.4e, resid: %.4e, iters: %i\n", tol, rsnew, i+1); */
			return i + 1;
		}
		ipzrOld = ipzr;
		applyPreConditioner(M, z, r, n, &ipzr);

		scaleArray(p, ipzr / ipzrOld, n);
		addScaledArray(p, z, n, 1);
	}
	return i;
}

/*  y += diag^-1*x */
static void accScaleDiag(idxint n, const pfloat *diag, const pfloat * x, pfloat * y) {
	for (idxint i = 0; i < n; i++) {
		y[i] += x[i]/diag[i];
	}
}


/* Zeros out tmp_n and tmp_m. */
static void resetTmp(Data * d, Priv * p) {
	memset(p->tmp_n, 0, d->n * sizeof(pfloat));
	memset(p->tmp_m, 0, d->m * sizeof(pfloat));
}

/*y = (RHO_X * I + A'A)x */
static void matVec(Data * d, Priv * p, const pfloat * x, pfloat * y) {
	pfloat * tmp = p->tmp_matvec;
	if isnan(x[0]) {
		return;
	}
	// printf("1. norm x %f\n", calcNorm(x, d->n));
	memset(tmp, 0, d->m * sizeof(pfloat));
	memset(y, 0, d->n * sizeof(pfloat));
	accumByA(d, p, x, tmp);
	// printf("2. norm tmp %f\n", calcNorm(tmp, d->m));
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
	resetTmp(d, p);
	PyObject* x_array;
	PyObject* y_array;
	if (d->NORMALIZE) {
		// tmp_m = D*x.
		accScaleDiag(d->m, d->D, x, p->tmp_m);
		// tmp_m *= SCALE.
		scaleArray(p->tmp_m, d->SCALE, d->m);
		x_array = vec_to_nparr(p->tmp_m, &(d->m));
		y_array = vec_to_nparr(p->tmp_n, &(d->n));
	} else {
		// tmp_m += SCALE*x.
		addScaledArray(p->tmp_m, x, d->SCALE, d->m);
		x_array = vec_to_nparr(p->tmp_m, &(d->m));
		y_array = vec_to_nparr(y, &(d->n));
	}
	PyObject *arglist;
	arglist = Py_BuildValue("(OO)", x_array, y_array);
	PyObject_CallObject(d->ATmul, arglist);
	Py_DECREF(arglist);

	// y += E*tmp_n.
	if (d->NORMALIZE) {
		accScaleDiag(d->n, d->E, p->tmp_n, y);
	}

	// AMatrix * A = d->A;
	// resetTmp(d, p);
	// accScaleDiag(d->m, d->D, x, p->tmp_m);
	// _accumByAtrans(d->n, A->x, A->i, A->p, p->tmp_m, p->tmp_n);
	// accScaleDiag(d->n, d->E, p->tmp_n, z);
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
	PyObject* x_array;
	PyObject* y_array;
	resetTmp(d, p);
	if (d->NORMALIZE) {
		// tmp_n = E*x.
		accScaleDiag(d->n, d->E, x, p->tmp_n);
		// tmp_n *= SCALE.
		scaleArray(p->tmp_n, d->SCALE, d->n);
		x_array = vec_to_nparr(p->tmp_n, &(d->n));
		y_array = vec_to_nparr(p->tmp_m, &(d->m));
	} else {
		// tmp_n += SCALE*x.
		addScaledArray(p->tmp_n, x, d->SCALE, d->n);
		x_array = vec_to_nparr(p->tmp_n, &(d->n));
		y_array = vec_to_nparr(y, &(d->m));
	}
	PyObject *arglist;
	arglist = Py_BuildValue("(OO)", x_array, y_array);
	PyObject_CallObject(d->Amul, arglist);
	Py_DECREF(arglist);

	// y += D*tmp_m.
	if (d->NORMALIZE) {
		accScaleDiag(d->m, d->D, p->tmp_m, y);
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
