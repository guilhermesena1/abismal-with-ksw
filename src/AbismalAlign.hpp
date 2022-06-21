/* Copyright (C) 2019 Andrew D. Smith
 *
 * Authors: Andrew D. Smith
 *
 * This file is part of ABISMAL.
 *
 * ABISMAL is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ABISMAL is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 */

#ifndef ABISMAL_ALIGN_HPP
#define ABISMAL_ALIGN_HPP

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream> //TODO: DELETE ME

// ksw
#include <stdlib.h>
#include <stdint.h>
#include <emmintrin.h>

#include "dna_four_bit.hpp" // genome_iterator
#include "cigar_utils.hpp"
#include "ksw.h"

// ksw
#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

typedef int16_t score_t;
typedef std::vector<uint8_t> Read; //4-bit encoding of reads
typedef genome_four_bit_itr genome_iterator;

// GS: this is needed both for ksw (so we don't have to pass it as
// args) but also in AbismalAlign
namespace aln_params {
  // alignment scores
  static const int8_t sa = 2; // match
  static const int8_t sb = 3; // mismatch
	static const int8_t gapo = 5; // gap open
  static const int8_t gape = 1; // gap extension
  static const int8_t max_off_diag = 50;

  // score matrix that matches query Ts with target Cs
  static const int8_t mat_c_to_t[] = {
     sa, -sb, -sb, -sb, -sb, //A -> A
    -sb,  sa, -sb,  sa, -sb, //C -> (C,T)
    -sb, -sb,  sa, -sb, -sb, //G -> G
    -sb, -sb, -sb,  sa, -sb, //T -> T
    -sb, -sb, -sb, -sb, -sb  //N -> nothing
  };

  // score matrix that matches query As with target Gs
  static const int8_t mat_g_to_a[] = {
     sa, -sb, -sb, -sb, -sb, //A -> A
    -sb,  sa, -sb, -sb, -sb, //C -> C
     sa, -sb,  sa, -sb, -sb, //G -> (A,G)
    -sb, -sb, -sb,  sa, -sb, //T -> T
    -sb, -sb, -sb, -sb, -sb  //N -> nothing
  };

  // in the target, bases are four bits per letter,
  // so we must convert
  // A = 0001 (1) -> 00 (0)
  // C = 0010 (2) -> 01 (1)
  // G = 0100 (4) -> 10 (2)
  // T = 1000 (8) -> 11 (3)
  // everything else is 4
  static const uint8_t
  fourbit_to_twobit_target[] = {4, 0, 1, 4, 2, 4, 4, 4, 3};

  // query is either T, rich or A-righ, so
  // we should also convert 1010 (Y) and 0101 (R)
  // A -> 0001 (1) -> 00 (0)
  // C -> 0010 (2) -> 01 (1)
  // G -> 0100 (4) -> 10 (2)
  // R -> 0101 (5) -> 00 (0)
  // T -> 1000 (8) -> 11 (3)
  // Y -> 1010 (10) -> 11 (3)
  static const uint8_t
  fourbit_to_twobit_query[] = {4, 0, 1, 4, 2, 0, 4, 4, 3, 4, 3};

  inline score_t max_single_score(const uint32_t readlen) {
    return sa*readlen;
  }

  inline score_t max_pair_score(const uint32_t len1, const uint32_t len2) {
    return sa*(len1+len2);
  }
};

/*************************************************
 *** BEGIN COPIED FROM ksw.c
 *************************************************/

const kswr_t g_defr = { 0, -1, -1, -1, -1};
struct _kswq_t {
	int qlen, slen;
	uint8_t shift, mdiff, max, size;
	__m128i *qp, *H0, *H1, *E, *Hmax;
};

/**
 * Initialize the query data structure
 *
 * @param size   Number of bytes used to store a score; valid valures are 1 or 2
 * @param qlen   Length of the query sequence
 * @param query  Query sequence
 * @param m      Size of the alphabet
 * @param mat    Scoring matrix in a one-dimension array
 *
 * @return       Query data structure
 */
kswq_t*
ksw_qinit(const int qlen, const uint8_t *query, const int8_t *mat) {
	static const int size = 1; // ksw uses this for 16 bit scores
  static const int m = 5; // alphabet size (ACGTN)
	static const int tmp = m*m; // shift
	static const int p = 8 * (3 - size); // # values per __m128i

	const int slen = (qlen + p - 1) / p; // segmented length

  int a; // iterator
	kswq_t *q;

	q = (kswq_t*)malloc(sizeof(kswq_t) + 256 + 16 * slen * (m + 4)); // a single block of memory
	q->qp = (__m128i*)(((size_t)q + sizeof(kswq_t) + 15) >> 4 << 4); // align memory
	q->H0 = q->qp + slen * m;
	q->H1 = q->H0 + slen;
	q->E  = q->H1 + slen;
	q->Hmax = q->E + slen;
	q->slen = slen; q->qlen = qlen; q->size = size;
	for (a = 0, q->shift = 127, q->mdiff = 0; a < tmp; ++a) {
    // find the minimum and maximum score
		if (mat[a] < (int8_t)q->shift) q->shift = mat[a];
		if (mat[a] > (int8_t)q->mdiff) q->mdiff = mat[a];
	}
	q->max = q->mdiff;
	q->shift = 256 - q->shift; // NB: q->shift is uint8_t
	q->mdiff += q->shift; // this is the difference between the min and max scores
	// An example: p=8, qlen=19, slen=3 and segmentation:
	//  {{0,3,6,9,12,15,18,-1},{1,4,7,10,13,16,-1,-1},{2,5,8,11,14,17,-1,-1}}
	if (size == 1) {
		int8_t *t = (int8_t*)q->qp;
		for (a = 0; a < m; ++a) {
			int i, k, nlen = slen * p;
			const int8_t *ma = mat + a * m;
			for (i = 0; i < slen; ++i)
				for (k = i; k < nlen; k += slen) // p iterations
					*t++ = (k >= qlen? 0 : ma[query[k]]) + q->shift;
		}
	}
  else {
		int16_t *t = (int16_t*)q->qp;
		for (a = 0; a < m; ++a) {
			int i, k, nlen = slen * p;
			const int8_t *ma = mat + a * m;
			for (i = 0; i < slen; ++i)
				for (k = i; k < nlen; k += slen) // p iterations
					*t++ = (k >= qlen? 0 : ma[query[k]]);
		}
	}
	return q;
}

void
print_result(const kswr_t &r) {
  std::cerr << r.score << " " << r.tb << " " << r.te << " " << r.qb << " " << r.qe <<"\n";
}

kswr_t
ksw_i16(const kswq_t *q, const int tlen, const uint8_t *target) {
   // the first gap costs -(_o+_e)
  static const int _gapo = aln_params::gapo;
  static const int _gape = aln_params::gape;
  static const int minsc = 0x10000;
	static const int endsc = 0x10000;
	const int slen = q->slen;

	int i, m_b, n_b, te = -1, gmax = 0;
  uint64_t *b;
	__m128i zero, gapoe, gape, *H0, *H1, *E, *Hmax;
	kswr_t r;

#define __max_8(ret, xx) do { \
		(xx) = _mm_max_epi16((xx), _mm_srli_si128((xx), 8)); \
		(xx) = _mm_max_epi16((xx), _mm_srli_si128((xx), 4)); \
		(xx) = _mm_max_epi16((xx), _mm_srli_si128((xx), 2)); \
    	(ret) = _mm_extract_epi16((xx), 0); \
	} while (0)

	// initialization
	r = g_defr;
	m_b = n_b = 0; b = 0;
	zero = _mm_set1_epi32(0);
	gapoe = _mm_set1_epi16(_gapo + _gape);
	gape = _mm_set1_epi16(_gape);
	H0 = q->H0; H1 = q->H1; E = q->E; Hmax = q->Hmax;
	for (i = 0; i < slen; ++i) {
		_mm_store_si128(E + i, zero);
		_mm_store_si128(H0 + i, zero);
		_mm_store_si128(Hmax + i, zero);
	}
	// the core loop
	for (i = 0; i < tlen; ++i) {
		int j, k, imax;
		__m128i e, h, f = zero, max = zero, *S = q->qp + target[i] * slen; // s is the 1st score vector
		h = _mm_load_si128(H0 + slen - 1); // h={2,5,8,11,14,17,-1,-1} in the above example
		h = _mm_slli_si128(h, 2);
		for (j = 0; LIKELY(j < slen); ++j) {
			h = _mm_adds_epi16(h, *S++);
			e = _mm_load_si128(E + j);
			h = _mm_max_epi16(h, e);
			h = _mm_max_epi16(h, f);
			max = _mm_max_epi16(max, h);
			_mm_store_si128(H1 + j, h);
			h = _mm_subs_epu16(h, gapoe);
			e = _mm_subs_epu16(e, gape);
			e = _mm_max_epi16(e, h);
			_mm_store_si128(E + j, e);
			f = _mm_subs_epu16(f, gape);
			f = _mm_max_epi16(f, h);
			h = _mm_load_si128(H0 + j);
		}
		for (k = 0; LIKELY(k < 16); ++k) {
			f = _mm_slli_si128(f, 2);
			for (j = 0; LIKELY(j < slen); ++j) {
				h = _mm_load_si128(H1 + j);
				h = _mm_max_epi16(h, f);
				_mm_store_si128(H1 + j, h);
				h = _mm_subs_epu16(h, gapoe);
				f = _mm_subs_epu16(f, gape);
				if(UNLIKELY(!_mm_movemask_epi8(_mm_cmpgt_epi16(f, h)))) goto end_loop8;
			}
		}
end_loop8:
		__max_8(imax, max);
		if (imax >= minsc) {
			if (n_b == 0 || (int32_t)b[n_b-1] + 1 != i) {
				if (n_b == m_b) {
					m_b = m_b? m_b<<1 : 8;
					b = (uint64_t*)realloc(b, 8 * m_b);
				}
				b[n_b++] = (uint64_t)imax<<32 | i;
			} else if ((int)(b[n_b-1]>>32) < imax) b[n_b-1] = (uint64_t)imax<<32 | i; // modify the last
		}
		if (imax > gmax) {
			gmax = imax; te = i;
			for (j = 0; LIKELY(j < slen); ++j)
				_mm_store_si128(Hmax + j, _mm_load_si128(H1 + j));
			if (gmax >= endsc) break;
		}
		S = H1; H1 = H0; H0 = S;
	}
	r.score = gmax; r.te = te;
  int max = -1, qlen = slen * 8;
  uint16_t *t = (uint16_t*)Hmax;
  for (i = 0, r.qe = -1; i < qlen; ++i, ++t)
    if ((int)*t > max) max = *t, r.qe = i / 8 + i % 8 * slen;

  free(b);
	return r;
}

kswr_t
ksw_u8(kswq_t *q, const int tlen, const uint8_t *target) { // the first gap costs -(_o+_e)
  // the first gap costs -(_o+_e)
  static const int _gapo = aln_params::gapo;
  static const int _gape = aln_params::gape;
  static const int minsc = 0x10000;
	static const int endsc = 0x10000;

	int slen, i, m_b, n_b, te = -1, gmax = 0;
	uint64_t *b;
	__m128i zero, gapoe, gape, shift, *H0, *H1, *E, *Hmax;
	kswr_t r;

#define __max_16(ret, xx) do { \
		(xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 8)); \
		(xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 4)); \
		(xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 2)); \
		(xx) = _mm_max_epu8((xx), _mm_srli_si128((xx), 1)); \
    	(ret) = _mm_extract_epi16((xx), 0) & 0x00ff; \
	} while (0)

	// initialization
	r = g_defr;
	m_b = n_b = 0; b = 0;
	zero = _mm_set1_epi32(0);
	gapoe = _mm_set1_epi8(_gapo + _gape);
	gape = _mm_set1_epi8(_gape);
	shift = _mm_set1_epi8(q->shift);
	H0 = q->H0; H1 = q->H1; E = q->E; Hmax = q->Hmax;
	slen = q->slen;
	for (i = 0; i < slen; ++i) {
		_mm_store_si128(E + i, zero);
		_mm_store_si128(H0 + i, zero);
		_mm_store_si128(Hmax + i, zero);
	}
	// the core loop
	for (i = 0; i < tlen; ++i) {
		int j, k, cmp, imax;
		__m128i e, h, f = zero, max = zero, *S = q->qp + target[i] * slen; // s is the 1st score vector
		h = _mm_load_si128(H0 + slen - 1); // h={2,5,8,11,14,17,-1,-1} in the above example
		h = _mm_slli_si128(h, 1); // h=H(i-1,-1); << instead of >> because x64 is little-endian
		for (j = 0; LIKELY(j < slen); ++j) {
			/* SW cells are computed in the following order:
			 *   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
			 *   E(i+1,j) = max{H(i,j)-q, E(i,j)-r}
			 *   F(i,j+1) = max{H(i,j)-q, F(i,j)-r}
			 */
			// compute H'(i,j); note that at the beginning, h=H'(i-1,j-1)
			h = _mm_adds_epu8(h, _mm_load_si128(S + j));
			h = _mm_subs_epu8(h, shift); // h=H'(i-1,j-1)+S(i,j)
			e = _mm_load_si128(E + j); // e=E'(i,j)
			h = _mm_max_epu8(h, e);
			h = _mm_max_epu8(h, f); // h=H'(i,j)
			max = _mm_max_epu8(max, h); // set max
			_mm_store_si128(H1 + j, h); // save to H'(i,j)
			// now compute E'(i+1,j)
			h = _mm_subs_epu8(h, gapoe); // h=H'(i,j)-gapo
			e = _mm_subs_epu8(e, gape); // e=E'(i,j)-gape
			e = _mm_max_epu8(e, h); // e=E'(i+1,j)
			_mm_store_si128(E + j, e); // save to E'(i+1,j)
			// now compute F'(i,j+1)
			f = _mm_subs_epu8(f, gape);
			f = _mm_max_epu8(f, h);
			// get H'(i-1,j) and prepare for the next j
			h = _mm_load_si128(H0 + j); // h=H'(i-1,j)
		}
		// NB: we do not need to set E(i,j) as we disallow adjecent insertion and then deletion
		for (k = 0; LIKELY(k < 16); ++k) { // this block mimics SWPS3; NB: H(i,j) updated in the lazy-F loop cannot exceed max
			f = _mm_slli_si128(f, 1);
			for (j = 0; LIKELY(j < slen); ++j) {
				h = _mm_load_si128(H1 + j);
				h = _mm_max_epu8(h, f); // h=H'(i,j)
				_mm_store_si128(H1 + j, h);
				h = _mm_subs_epu8(h, gapoe);
				f = _mm_subs_epu8(f, gape);
				cmp = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_subs_epu8(f, h), zero));
				if (UNLIKELY(cmp == 0xffff)) goto end_loop16;
			}
		}
end_loop16:
		//int k;for (k=0;k<16;++k)printf("%d ", ((uint8_t*)&max)[k]);printf("\n");
		__max_16(imax, max); // imax is the maximum number in max
		if (imax >= minsc) { // write the b array; this condition adds branching unfornately
			if (n_b == 0 || (int32_t)b[n_b-1] + 1 != i) { // then append
				if (n_b == m_b) {
					m_b = m_b? m_b<<1 : 8;
					b = (uint64_t*)realloc(b, 8 * m_b);
				}
				b[n_b++] = (uint64_t)imax<<32 | i;
			} else if ((int)(b[n_b-1]>>32) < imax) b[n_b-1] = (uint64_t)imax<<32 | i; // modify the last
		}
		if (imax > gmax) {
			gmax = imax; te = i; // te is the end position on the target
			for (j = 0; LIKELY(j < slen); ++j) // keep the H1 vector
				_mm_store_si128(Hmax + j, _mm_load_si128(H1 + j));
			if (gmax + q->shift >= 255 || gmax >= endsc) break;
		}
		S = H1; H1 = H0; H0 = S; // swap H0 and H1
	}
	r.score = gmax + q->shift < 255? gmax : 255;
	r.te = te;
	if (r.score != 255) { // get a->qe, the end of query match; find the 2nd best score
		int max = -1,  qlen = slen * 16;
		uint8_t *t = (uint8_t*)Hmax;
		for (i = 0; i < qlen; ++i, ++t)
			if ((int)*t > max) max = *t, r.qe = i / 16 + i % 16 * slen;
		//printf("%d,%d\n", max, gmax);
		if (b) {
			i = (r.score + q->max - 1) / q->max;
		}
	}
	free(b);
	return r;
}
static void
revseq(int l, uint8_t *s) {
	int i, t;
	for (i = 0; i < l>>1; ++i)
		t = s[i], s[i] = s[l - 1 - i], s[l - 1 - i] = t;
}

kswr_t
ksw_align(const int qlen, const int tlen,
          const int8_t *mat,
          uint8_t *query, uint8_t *target, kswq_t **qry) {
	//kswr_t r = ksw_i16(*qry, tlen, target);
	kswr_t r = ksw_u8(*qry, tlen, target);

  // full alignment: find the start position by alignment of revcomps
  // NB: reverse is *NOT* the reverse-complement, so mat should be
  // the same in both directions (i.e. no bisulfite base flip)
  revseq(r.qe + 1, query); revseq(r.te + 1, target);

  kswq_t *q = ksw_qinit(r.qe + 1, query, mat);
  //kswr_t rr = ksw_i16(q, r.te + 1, target);
  kswr_t rr = ksw_u8(q, r.te + 1, target);
  // GS: maybe we don't need this if ksw_align is only
  // going to be called once?
  revseq(r.qe + 1, query); revseq(r.te + 1, target);

  if (r.score == rr.score)
    r.tb = r.te - rr.te, r.qb = r.qe - rr.qe;

  assert(r.tb >= 0);
  assert(r.qb >= 0);
  free(q);
	return r;
}

/********************
 * Global alignment *
 ********************/

#define MINUS_INF -0x40000000
typedef struct {
  int32_t h, e;
} eh_t;

static inline uint32_t*
push_cigar(int *n_cigar, int *m_cigar, uint32_t *cigar,
           const uint32_t op, const int len) {
	if (*n_cigar == 0 || op != (cigar[(*n_cigar) - 1]&0xf)) {
		if (*n_cigar == *m_cigar) {
			*m_cigar = *m_cigar? (*m_cigar)<<1 : 4;
			cigar = (uint32_t*) realloc(cigar, (*m_cigar) << 2);
		}
		cigar[(*n_cigar)++] = len<<4 | op;
	}
  else cigar[(*n_cigar)-1] += len<<4;
	return cigar;
}

int
ksw_global(const int qlen, const uint8_t *query,
           const int tlen, const uint8_t *target,
           const int8_t *mat, const int w,
           int *n_cigar_, uint32_t **cigar_) {

  static const int m = 5;
  static const int gapo = aln_params::gapo;
  static const int gape = aln_params::gape;

  const int gapoe = gapo + gape;
	const int n_col = qlen < 2*w+1? qlen : 2*w+1;
	eh_t *eh;
	int8_t *qp; // query profile
	int i;
  int j;
  int k;
  int score;

	// backtrack matrix; in each cell: f<<4|e<<2|h;
  // in principle, we can halve the memory,
  // but backtrack will be a little more complex
  uint8_t *z;

	if (n_cigar_)
    *n_cigar_ = 0;

	// allocate memory

  // maximum #columns of the backtrack matrix
  z = (uint8_t*) malloc(n_col * tlen);
	qp = (int8_t*) malloc(qlen * m);
	eh = (eh_t*) calloc(qlen + 1, 8);

	// generate the query profile
	for (k = i = 0; k < m; ++k) {
		const int8_t *p = &mat[k * m];
		for (j = 0; j < qlen; ++j) qp[i++] = p[query[j]];
	}

	// fill the first row
	eh[0].h = 0; eh[0].e = MINUS_INF;
	for (j = 1; j <= qlen && j <= w; ++j)
		eh[j].h = -(gapo + gape * j), eh[j].e = MINUS_INF;

	for (; j <= qlen; ++j)
    eh[j].h = eh[j].e = MINUS_INF; // everything is -inf outside the band

	// DP loop
	for (i = 0; LIKELY(i < tlen); ++i) { // target sequence is in the outer loop
		int32_t f = MINUS_INF;
    int32_t h1;
    int32_t beg;
    int32_t end;
		int8_t *q = &qp[target[i] * qlen];
		uint8_t *zi = &z[i * n_col];
		beg = i > w? i - w : 0;
		end = i + w + 1 < qlen? i + w + 1 : qlen; // only loop through [beg,end) of the query sequence
		h1 = beg == 0? -(gapo + gape * (i + 1)) : MINUS_INF;
		for (j = beg; LIKELY(j < end); ++j) {
			// This loop is organized in a similar way to ksw_extend() and ksw_sse2(), except:
			// 1) not checking h>0; 2) recording direction for backtracking
			eh_t *p = &eh[j];
			int32_t h = p->h, e = p->e;
			uint8_t d; // direction
			p->h = h1;
			h += q[j];
			d = h > e? 0 : 1;
			h = h > e? h : e;
			d = h > f? d : 2;
			h = h > f? h : f;
			h1 = h;
			h -= gapoe;
			e -= gape;
			d |= e > h? 1<<2 : 0;
			e = e > h? e : h;
			p->e = e;
			f -= gape;
			d |= f > h? 2<<4 : 0; // if we want to halve the memory, use one bit only, instead of two
			f = f > h? f : h;
			zi[j - beg] = d; // z[i,j] keeps h for the current cell and e/f for the next cell
		}
		eh[end].h = h1; eh[end].e = MINUS_INF;
	}

	score = eh[qlen].h;
	if (n_cigar_ && cigar_) { // backtrack
		int n_cigar = 0, m_cigar = 0, which = 0;
		uint32_t *cigar = 0, tmp;
		i = tlen - 1; k = (i + w + 1 < qlen? i + w + 1 : qlen) - 1; // (i,k) points to the last cell
		while (i >= 0 && k >= 0) {
			which = z[i * n_col + (k - (i > w? i - w : 0))] >> (which<<1) & 3;
			if (which == 0)      cigar = push_cigar(&n_cigar, &m_cigar, cigar, 0, 1), --i, --k;
			else if (which == 1) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 2, 1), --i;
			else                 cigar = push_cigar(&n_cigar, &m_cigar, cigar, 1, 1), --k;
		}
		if (i >= 0) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 2, i + 1);
		if (k >= 0) cigar = push_cigar(&n_cigar, &m_cigar, cigar, 1, k + 1);
		for (i = 0; i < n_cigar>>1; ++i) // reverse CIGAR
			tmp = cigar[i], cigar[i] = cigar[n_cigar-1-i], cigar[n_cigar-1-i] = tmp;
		*n_cigar_ = n_cigar, *cigar_ = cigar;
	}
	free(eh); free(qp); free(z);
	return score;
}

/*************************************************
 *** END COPIED FROM ksw.c
 *************************************************/

struct AbismalAlign {
  // frees memory for new read
  AbismalAlign(const genome_iterator &target_start);

  // encodes four-bit reads in two bits
  void reset(const uint32_t readlen);

  // populates either enc_query_rc or enc_query
  void encode_query(const bool rc, const Read &pread);

  // populates enc_target with reference genome
  // sequence starting at t_pos
  void encode_target(const uint32_t t_pos, const uint32_t t_sz);

  // align without traceback, use ksw_i16
  score_t ksw_score(const score_t diffs, const bool rc, const bool a_rich, const uint32_t t_pos);

  // align with traceback, populate r, n_cigar and cigar
  bool ksw_full(const bool rc, const bool a_rich,
                const score_t diffs, const score_t max_diffs,
                const uint32_t t_pos);

  // "translates" n_cigar and cigar32 to human-readable
  void inflate_cigar(std::string &cigar) const;

  // converts the results of ksw_full to an edit distance
  inline score_t edit_distance() const;

  inline uint32_t get_query_aln_length() const;

  // populates n_cigar and cigar32 with a series of matches.
  // e.g. if readlen = 100, CIGAR becomes 100M
  void make_default_cigar();

  /************** INLINE HELPER FUNCTIONS **************/
  // return either enc_query_rc or enc_query
  inline uint8_t* get_query(const bool rc) const;

  // populates the query profile for alignment
  inline void set_query_profile(const bool rc, const bool a_rich);

  /************** STRUCT DATA **************/
  // query profiles for A/T-rich, read or revcomp
  kswq_t* q_t;
  kswq_t* q_t_rc;
  kswq_t* q_a;
  kswq_t* q_a_rc;

  // these variables are populated at reset()
  size_t q_sz; // query size
  uint8_t *enc_query; // query in two bits
  uint8_t *enc_query_rc; // rc of query in two bits

  // this is populated at alignments, either ksw_score or ksw_full
  uint8_t* enc_target;

  // this will be reported in the SAM entry. Populated in ksw_full
  int* n_cigar; // number of CIGAR ops
  uint32_t* cigar32; // array of CIGAR ops
  kswr_t r; // score, target+query start+end

  const genome_iterator target;
};

// constructor: allocate pointers that will be reused
AbismalAlign::AbismalAlign(const genome_iterator &target_start) :
  target(target_start) {
  static const uint32_t MAX_READ_LENGTH = 1024;

  enc_target =    (uint8_t*) malloc(MAX_READ_LENGTH);
  enc_query =     (uint8_t*) malloc(MAX_READ_LENGTH);
  enc_query_rc =  (uint8_t*) malloc(MAX_READ_LENGTH);
  n_cigar =       (int*)     malloc(sizeof(int));

  q_t = q_a = q_t_rc = q_a_rc = NULL;
  cigar32 = NULL;
}

// unpacks some genome subsequence to two bits per nt
// NB: genome is a vector<size_t>, with 16 letters per size_t element
void
AbismalAlign::encode_target(const uint32_t t_beg, const uint32_t tlen) {
  genome_iterator t_itr = target + t_beg;
  uint8_t *itr = enc_target;
  for (uint32_t i = 0; i < tlen; ++i, ++t_itr, ++itr)
    *itr = aln_params::fourbit_to_twobit_target[*t_itr];
}

// unpacks query sequence to two bits per nt
// input is a vector with four-bit bases
void
AbismalAlign::encode_query(const bool rc, const Read &pread) {
  const auto lim = std::end(pread);
  uint8_t *itr = get_query(rc);
  for (auto it(begin(pread)); it != lim; ++it, ++itr)
    *itr = aln_params::fourbit_to_twobit_query[*it];
}


// prepare current read for alignment.
void
AbismalAlign::reset(const uint32_t readlen) {
  q_sz = readlen;
  r = {0, static_cast<int>(readlen -  1), static_cast<int>(readlen - 1), 0, 0};

  // deletes previous query profile
  free(q_t); free(q_t_rc); free(q_a); free(q_a_rc);
  q_t = q_t_rc = q_a = q_a_rc = NULL;
}

// populates a query profile that will be needed for alignment,
// if it hasn't already been done.
inline void
AbismalAlign::set_query_profile(const bool rc, const bool a_rich) {
  if (a_rich) {
    if (rc && q_a_rc == NULL) {
      q_a_rc  = ksw_qinit(q_sz, enc_query_rc, aln_params::mat_g_to_a);
    }
    if (!rc && q_a == NULL) {
      q_a  = ksw_qinit(q_sz, enc_query, aln_params::mat_g_to_a);
    }
  }
  else {
    if (rc && q_t_rc == NULL) {
      q_t_rc = ksw_qinit(q_sz, enc_query_rc, aln_params::mat_c_to_t);
    }
    if (!rc && q_t == NULL) {
      q_t = ksw_qinit(q_sz, enc_query, aln_params::mat_c_to_t);
    }
  }
}

// Smith-Waterman without traceback
score_t
AbismalAlign::ksw_score(const score_t diffs, const bool rc, const bool a_rich, const uint32_t t_pos) {
  if (diffs == 0)
    make_default_cigar();

  // lazy encoding to avoid this work if read is never aligned
  set_query_profile(rc, a_rich);
  encode_target(t_pos, q_sz); // NB: here t_sz = q_sz. Should we use bandwidth?
  return static_cast<score_t>(
    ksw_u8(
    //ksw_i16(
      rc ? (a_rich ? q_a_rc : q_t_rc) : (a_rich ? q_a : q_t),
      q_sz, enc_target).score
  );
}

template<class T> inline T
min16(const T a, const T b) {
  return (a<b) ? a:b;
}

inline uint8_t*
AbismalAlign::get_query(const bool rc) const {
  return rc ? enc_query_rc : enc_query;
}

// Smith-Waterman with traceback and CIGAR
bool
AbismalAlign::ksw_full(const bool rc, const bool a_rich,
                       const score_t diffs, const score_t max_diffs,
                       const uint32_t t_pos) {
  /*const uint32_t bw = min16(
      static_cast<uint32_t>(2*min16(diffs, max_diffs) + 1),
      static_cast<uint32_t>(max_off_diag);*/
  const uint32_t bw = aln_params::max_off_diag;
  const uint32_t t_sz = q_sz; // add bandwidth around target?
                              //
  encode_target(t_pos, t_sz);

  // ksw_align finds both the score and the start and end of the
  // local alignment by also aligning the revcomp of the
  // target and the query
  r = ksw_align(t_sz, q_sz,
        a_rich ? aln_params::mat_g_to_a : aln_params::mat_c_to_t,
        get_query(rc), enc_target,
        rc ? (a_rich ? &q_a_rc : &q_t_rc) : (a_rich ? &q_a : &q_t)
      );

  if (r.score < 0) return false;

  // banded global alignment with bounds devised by ksw_align
  r.score = ksw_global(
    r.qe - r.qb + 1, get_query(rc) + r.qb,
    r.te - r.tb + 1, enc_target + r.tb,
    a_rich ? aln_params::mat_g_to_a : aln_params::mat_c_to_t,
    bw, n_cigar, &cigar32
  );
  if (r.score < 0) return false;

  return true;
}

inline uint32_t
get_num_cigar_op(const uint32_t cigar_op) {
  return (cigar_op >> 4);
}

inline char
get_which_cigar_op(const uint32_t cigar_op) {
  return "MIDSPH"[cigar_op&0xf];
}

void
AbismalAlign::inflate_cigar(std::string &cigar) const {
  std::ostringstream oss;
  if (r.qb > 0) // soft-clip left
    oss << r.qb << 'S';

  uint32_t num_ops;
  char op;
  score_t qseq_ops = r.qb;
  for (int i = 0; i < *n_cigar; ++i) {
    num_ops = get_num_cigar_op(cigar32[i]);
    op = get_which_cigar_op(cigar32[i]);
    oss << num_ops << op;
    qseq_ops += consumes_query(op)*num_ops;
  }

  if (qseq_ops < static_cast<score_t>(q_sz)) // soft-clip right
    oss << (q_sz - qseq_ops) << 'S';
  cigar = oss.str();

  if (cigar_qseq_ops(cigar) != q_sz) {
    std::cerr << " * ERROR: failing at inconsistent qseq ops\n";
    std::cerr << "cigar: " << cigar << "\n";
    std::cerr << "qseq ops: " << cigar_qseq_ops(cigar) << "\n";
    std::cerr << "r: " << r.score << " " << r.tb << " " << r.te << " " << r.qb << " " << r.qe << "\n";
    std::cerr << "query size: " << q_sz << "\n";
    assert(false);
  }
}

score_t
AbismalAlign::edit_distance() const {
  std::string s;
  inflate_cigar(s);
  if (r.score <= 0)
    return std::numeric_limits<score_t>::max();

  score_t score_all_matches = 0;
  score_t num_edits = 0;

  uint32_t op;
  uint32_t num_ops;
  for (int i = 0; i < *n_cigar; ++i) {
    op = (cigar32[i] & 0xF); //0, 1 or 2
    assert(op <= 2);
    num_ops = get_num_cigar_op(cigar32[i]);

    if (op == 1 || op == 2) {// I or D
      num_edits += num_ops;
      score_all_matches -= aln_params::gapo + num_ops*aln_params::gape;
    }
    else if (op == 0)
      score_all_matches += num_ops*aln_params::sa;
  }
  score_all_matches -= r.score;


  assert(score_all_matches%(aln_params::sa + aln_params::sb) == 0);
  // num mismatches as a function of the score and the score with all
  // matche
  const score_t num_mismatches =
    score_all_matches/(aln_params::sa + aln_params::sb);
  num_edits += num_mismatches;

  return num_edits;
}

inline uint32_t
AbismalAlign::get_query_aln_length() const {
  return r.qe - r.qb + 1;
}

void
AbismalAlign::make_default_cigar() {
  *n_cigar = 1;
  cigar32 = (uint32_t*)malloc(sizeof(uint32_t));

  // puts readlen in the first 28 bits, leave the last 4
  // bits as 0 as the "match" (M) op is 0000
  cigar32[0] = (q_sz << 4);
}

#endif


