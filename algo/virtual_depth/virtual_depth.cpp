
/*

 * @Author: Qing Hong

 * @Date: 2023-05-12 16:59:39

 * @LastEditors: QingHong

 * @LastEditTime: 2023-05-24 17:38:32

 * @Description: file content

 */

#include "stdlib.h"

#include<string.h>

typedef float DTYPE;

#define __int64 long long

#define        MIN(a,b)        ((a)>(b)?(b):(a))

#define        MAX(a,b)        ((a)>(b)?(a):(b))

#define     ABS(a)          ((a)>=0? (a):(-(a)))

template<typename T> const T& min(const T& a, const T& b) { return a < b ? a : b; }

template<typename T> const T& max(const T& a, const T& b) { return a > b ? a : b; }

// typedef struct _tagFrameData{

//     DTYPE r;

//     DTYPE g;

//     DTYPE b;

//     DTYPE a;

// }FrameData;

typedef struct _tagMVD{

    DTYPE x;

    DTYPE y;

    DTYPE z;

    DTYPE depth;

    DTYPE alpha;

    bool fb_flag;

    int  obj_mask;

    

}MVD;

//motion vector for global fg/bg MV, 96bits

typedef struct _tagMVector_GlbOccMV{

    int bg_x;           //s10bits

    int bg_y;           //s10bits

    int fg_x;           //s10bits

    int fg_y;           //s10bits

    

  //  int dcratio;        // 8bits

  //  int sceneflag;      // 1bit

   // int flag;           // 8bits

  //  int fgcnt;          // 10bits

  //  int bgcnt;          // 10bits

    

    int w_fgbg;

    

} MVector_GlbOccMV;

typedef double DTYPEIM;   // image data type

typedef struct _tagMVector_GMV{

    float x;

    float y;

    float depth;

    bool lab;

    bool occ_lab;

    double mvdiff;   // for 0/1 phase, it's the difference bwteween fg/bg

    int sad;

        

    float w_fgbg;

    

    bool bk_en;  // if lab_bk=1, means there is fg/bg

    float fg_x;   // if bk

    float fg_y;

    float fg_d;

        

    float bg_x;   // if bk

    float bg_y;

    float bg_d;

    

    bool fg_lab;

    int bk_cnt;

    

    int hitcnt;

    

    float depth_d;

} MVector_GMV;

typedef struct _tagFilmBorder{

    int StartRow;

    int EndRow;

    int StartCol;

    int EndCol;

} FmBdInfo;

void trans(float* data,MVD* result,int nHeight,int nWidth,int mv0);

bool IsInFmbd(int row, int col, FmBdInfo fmbd);

void GMVHitPartner(MVector_GMV *pMV_io, MVector_GMV *pMV_ref, double nWeight, int nBlkSize, int nHNum, int nVNum, FmBdInfo fmbd_blk, MVector_GMV *pMV_partner);

void GMVFGBGDet(MVector_GMV *pMV0, MVector_GMV *pMV1, MVector_GMV *pMV0_hit1, MVector_GMV *pMV1_hit0, int nWidth_Im, int nHeight_Im, FmBdInfo fmbd_i, MVector_GlbOccMV *GlbOccMV_o);

void VirtualDepthCalc_Core(MVector_GMV *pMV0, MVector_GMV *pMV1, MVector_GlbOccMV *GlbOccMV_i, int nHNum, int nVNum, FmBdInfo fmbd_blk);

void Depth_Interpolation(MVector_GMV *pMV_i, int nBlkSize, float w_Gain, int nWidth, int nHeight, FmBdInfo fmbd_i, float *pDepth, float *pGain);

void trans(float* data,MVD* result,int nHeight,int nWidth,int mv0){

     for(int nRow = 0; nRow < nHeight; nRow++)

        {

            for(int nCol = 0; nCol < nWidth; nCol++)

            {

                int nIndex = nCol + nRow * nWidth;

                float xx = (float)data[nIndex*2]*mv0;

                float yy = (float)data[nIndex*2+1];

                result[nIndex].x = xx;

                result[nIndex].y = yy;

        }

}

}

bool IsInFmbd(int row, int col, FmBdInfo fmbd)

{

    bool bInFmbd= (row>=fmbd.StartRow && row<fmbd.EndRow && col>=fmbd.StartCol && col<fmbd.EndCol);

    return bInFmbd;

}

void GMVHitPartner(MVector_GMV *pMV_io, MVector_GMV *pMV_ref, double nWeight, int nBlkSize, int nHNum, int nVNum, FmBdInfo fmbd_blk, MVector_GMV *pMV_partner)

{

    int nMVSubBits=0;

    int nDiv = nBlkSize*(1<<nMVSubBits);

    int nMVSize = nVNum*nHNum;

    

    MVector_GMV *pFIMV_t= new MVector_GMV[nMVSize]; memset(pFIMV_t, 0, nMVSize*sizeof( MVector_GMV));

    for (int kk=0; kk<nMVSize; kk++)    pMV_io[kk].hitcnt=0;

    //======== 1  hit to ref, and get pMV_partner, and keep the one closes to pMV_ref

    int blkidx=0;

    for (int vb=0; vb<nVNum; vb++) for (int hb=0; hb<nHNum; hb++, blkidx++)

    {

        if (IsInFmbd(vb, hb, fmbd_blk))

        {

            MVector_GMV mv_in = pMV_io[blkidx];

            

            int hb_hit = ((int)(hb*64 + mv_in.x*nWeight/nDiv + 32))>>6;

            int vb_hit = ((int)(vb*64 + mv_in.y*nWeight/nDiv + 32))>>6;

            

            if (IsInFmbd(vb_hit, hb_hit, fmbd_blk))

            {

                int idx_hit = vb_hit*nHNum + hb_hit;

                MVector_GMV mv_hit = pMV_partner[idx_hit];

                MVector_GMV mv_ref = pMV_ref[idx_hit];

                int mvdiff_in2ref  = ABS(mv_ref.x - mv_in.x) + ABS(mv_ref.y - mv_in.y);

                int mvdiff_hit2ref = ABS(mv_ref.x - mv_hit.x) + ABS(mv_ref.y - mv_hit.y);

                

                pMV_partner[idx_hit].hitcnt++;

                

                if ( mv_hit.lab==0 || (mv_hit.lab && (mvdiff_in2ref<mvdiff_hit2ref)))

                {

                    pMV_partner[idx_hit].x   = mv_in.x;

                    pMV_partner[idx_hit].y   = mv_in.y;

                    pMV_partner[idx_hit].lab=1;

                }

            }

        }

    }

        

    // 2 hit pMV_partner back to pMV_i and get pFIMV_o

    blkidx=0;

    for (int vb=0; vb<nVNum; vb++) for (int hb=0; hb<nHNum; hb++, blkidx++)

    {

        if (IsInFmbd(vb, hb, fmbd_blk))

        {

            MVector_GMV mv_in = pMV_partner[blkidx];

            

            int hb_hit = ((int)(hb*64 - mv_in.x*nWeight/nDiv + 32))>>6;

            int vb_hit = ((int)(vb*64 - mv_in.y*nWeight/nDiv + 32))>>6;

            

            if (IsInFmbd(vb_hit, hb_hit, fmbd_blk))

            {

                int idx_hit = vb_hit*nHNum + hb_hit;

                MVector_GMV mv_hit = pFIMV_t[idx_hit];

                MVector_GMV mv_ref = pMV_io[idx_hit];

                int mvdiff_in2ref  = ABS(mv_ref.x - mv_in.x ) + ABS(mv_ref.y - mv_in.y );

                int mvdiff_hit2ref = ABS(mv_ref.x - mv_hit.x) + ABS(mv_ref.y - mv_hit.y);

                if ( mv_hit.lab==0 || (mv_hit.lab && (mvdiff_in2ref<mvdiff_hit2ref)))

                {

                    pFIMV_t[idx_hit].x   = mv_in.x;

                    pFIMV_t[idx_hit].y   = mv_in.y;

                    pFIMV_t[idx_hit].lab=1;

                    pFIMV_t[idx_hit].hitcnt = mv_in.hitcnt;

                    

                }

            }

        }

    }

    

    // 3. hit pMV_io to pMV_partner to get mvdiff

    blkidx=0;

    for (int vb=0; vb<nVNum; vb++) for (int hb=0; hb<nHNum; hb++, blkidx++)

    {

        if (IsInFmbd(vb, hb, fmbd_blk))

        {

            MVector_GMV mv_in = pMV_io[blkidx];

            

            int hb_hit = ((int)(hb*64 + mv_in.x*nWeight/nDiv + 32))>>6;

            int vb_hit = ((int)(vb*64 + mv_in.y*nWeight/nDiv + 32))>>6;

            

            if (IsInFmbd(vb_hit, hb_hit, fmbd_blk))

            {

                int idx_hit = vb_hit*nHNum + hb_hit;

                MVector_GMV mv_hit = pMV_partner[idx_hit];

                int mvdiff = ABS(mv_in.x - mv_hit.x) + ABS(mv_in.y - mv_hit.y);

                pMV_io[blkidx].mvdiff = mvdiff;

            }

        }

    }

    

    

    // 3 reduce wrong detect hitcnt:

    int *pHitCnt= new int[nMVSize];     for (int kk=0; kk<nMVSize; kk++) { pHitCnt[kk] = pFIMV_t[kk].hitcnt;}

    

    blkidx=0;  int ws=3;

    for (int vb=0; vb<nVNum; vb++) for (int hb=0; hb<nHNum; hb++, blkidx++)

    {

        int hitcnt=pHitCnt[blkidx];

        if (IsInFmbd(vb, hb, fmbd_blk) && vb>=ws && vb<(nVNum-ws) && hb>=ws && hb<(nHNum-ws))

        {

            if (hitcnt>1 || hitcnt<1)

            {

                int cnt_0 = 0, cnt_2=0, cnt_1=0;

                for (int ii=-ws; ii<=ws; ii++)  for (int jj=-ws; jj<=ws; jj++)

                {

                    int idx_t = (vb + ii)*nHNum + hb + jj;

                    

                    if     (pHitCnt[idx_t]==0)   cnt_0++;

                    else if(pHitCnt[idx_t]>=2)   cnt_2++;

                    else                         cnt_1++;

                }

                

                if (cnt_1>(cnt_0 + cnt_2))

                {

                    if (hitcnt>1 && cnt_0>0) hitcnt = 1;

                    if (hitcnt<1 && cnt_2>0) hitcnt = 1;

                }

            }

        }

        

        pMV_io[blkidx].hitcnt = hitcnt;

    }

    

            

    delete []pHitCnt;       pHitCnt=NULL;

    delete []pFIMV_t;       pFIMV_t=NULL;

}

void VirtualDepthCalc_Top(MVD *pMVD0_io, MVD *pMVD1_io, int nWidth, int nHeight, FmBdInfo fmbd)

{

    int nBlkSize=1;

    int nImSize = nHeight*nWidth;

    

    MVector_GlbOccMV GlbOccMV_o; 

    memset(&GlbOccMV_o, 0, sizeof(MVector_GlbOccMV));

 //   FmBdInfo fmbd; fmbd.StartCol=0; fmbd.EndCol=nWidth; fmbd.StartRow=0; fmbd.EndRow=nHeight;

    

    DTYPEIM *pVidP1[3]; for (int cc=0; cc<3; cc++) { pVidP1[cc]=new DTYPEIM[nImSize]; }

    DTYPEIM *pVidCF[3]; for (int cc=0; cc<3; cc++) { pVidCF[cc]=new DTYPEIM[nImSize]; }

    

    MVector_GMV *pMVD0 = new MVector_GMV[nImSize]; memset(pMVD0, 0, nImSize*sizeof(MVector_GMV));

    MVector_GMV *pMVD1 = new MVector_GMV[nImSize]; memset(pMVD1, 0, nImSize*sizeof(MVector_GMV));

    

    for (int kk=0; kk<nImSize; kk++)

    {

        pMVD0[kk].x =-pMVD0_io[kk].x;       pMVD0[kk].y =-pMVD0_io[kk].y;   pMVD0[kk].depth = 0;//pMVD0_io[kk].depth;

        pMVD1[kk].x = pMVD1_io[kk].x;       pMVD1[kk].y = pMVD1_io[kk].y;   pMVD1[kk].depth = 0;//pMVD1_io[kk].depth;

        

    //    pVidP1[0][kk] = m_pP1Data[kk].r;  pVidP1[1][kk] = m_pP1Data[kk].g;  pVidP1[2][kk] = m_pP1Data[kk].b;

    //    pVidCF[0][kk] = m_pCFData[kk].r;  pVidCF[1][kk] = m_pCFData[kk].g;  pVidCF[2][kk] = m_pCFData[kk].b;

    }

    

    MVector_GMV *pMVD0_hit1 = new MVector_GMV[nImSize]; memset(pMVD0_hit1, 0, nImSize*sizeof(MVector_GMV));

    MVector_GMV *pMVD1_hit0 = new MVector_GMV[nImSize]; memset(pMVD1_hit0, 0, nImSize*sizeof(MVector_GMV));

    GMVHitPartner(pMVD0, pMVD1, -64,  1, nWidth, nHeight, fmbd, pMVD0_hit1);

    GMVHitPartner(pMVD1, pMVD0,  64,  1, nWidth, nHeight, fmbd, pMVD1_hit0);

    

    GMVFGBGDet(pMVD0, pMVD1, pMVD0_hit1, pMVD1_hit0, nWidth, nHeight, fmbd, &GlbOccMV_o);

    

    VirtualDepthCalc_Core(pMVD0, pMVD1, &GlbOccMV_o, nWidth, nHeight, fmbd);

    

    int kk=0;

    for (int row=0; row<nHeight; row++) for (int col=0; col<nWidth; col++, kk++)

    {

        if (IsInFmbd(row, col, fmbd))

        {

            pMVD0_io[kk].depth= pMVD0[kk].depth;

            pMVD1_io[kk].depth = pMVD1[kk].depth;

        }

        else

        {

            pMVD0_io[kk].x= 0; pMVD0_io[kk].y= 0; pMVD0_io[kk].depth = 0;

            pMVD1_io[kk].x= 0; pMVD1_io[kk].y= 0; pMVD1_io[kk].depth = 0;

        }

    }

    

    delete []pMVD0; pMVD0=NULL;

    delete []pMVD1; pMVD1=NULL;

    

    delete []pMVD0_hit1;  pMVD0_hit1=NULL;

    delete []pMVD1_hit0;  pMVD1_hit0=NULL;

    for (int cc=0; cc<3; cc++) { delete [](pVidP1[cc]);  pVidP1[cc]=NULL; }

    for (int cc=0; cc<3; cc++) { delete [](pVidCF[cc]);  pVidCF[cc]=NULL; }

}

void GMVFGBGDet(MVector_GMV *pMV0, MVector_GMV *pMV1, MVector_GMV *pMV0_hit1, MVector_GMV *pMV1_hit0, int nWidth_Im, int nHeight_Im, FmBdInfo fmbd_i, MVector_GlbOccMV *GlbOccMV_o)

{

    int nBlkSize=1;

    int nMVSubBits=0;

    double nMVDiv = nBlkSize*(1<<0);

    

    int nHNum = nWidth_Im /nBlkSize;

    int nVNum = nHeight_Im/nBlkSize;

    int nMVSize = nVNum*nHNum;

    FmBdInfo fmbd_blk;

    fmbd_blk.StartCol = fmbd_i.StartCol/nBlkSize;

    fmbd_blk.StartRow = fmbd_i.StartRow/nBlkSize;

    fmbd_blk.EndCol   = fmbd_i.EndCol/nBlkSize;

    fmbd_blk.EndRow   = fmbd_i.EndRow/nBlkSize;

    

    

    for (int iter=0; iter<4; iter++)

    {

        MVector_GMV *pMV_t = (iter==0)? pMV0 : ((iter==1)? pMV1 : ((iter==2)? pMV0_hit1 : pMV1_hit0));

        for (int kk=0; kk<nMVSize; kk++) { pMV_t[kk].bg_x=0; pMV_t[kk].bg_y=0; pMV_t[kk].fg_x=0; pMV_t[kk].fg_y=0; pMV_t[kk].w_fgbg=0; }

    }

    

    // get the fgbg mv

    for (int iter=0; iter<2; iter++)

    {

        int nSign = (iter==0)? (-1) : 1;

        MVector_GMV *pMV_cur = (iter==0)? pMV0 : pMV1;

        MVector_GMV *pMV_hit = (iter==0)? pMV0_hit1 : pMV1_hit0;

        

        for (int vb=0; vb<nVNum; vb++) for (int hb=0; hb<nHNum; hb++)

        {

            int blkidx = vb*nHNum + hb;

            bool bInFmBd = IsInFmbd(vb, hb, fmbd_blk);

            MVector_GMV mv_i = pMV_cur[blkidx];

            

            if (mv_i.hitcnt==0)

            {

                MVector_GMV mv_bg = mv_i;

                

                int vb_hit0 = ((int)(vb + nSign*mv_bg.y/nMVDiv+0.5));

                int hb_hit0 = ((int)(hb + nSign*mv_bg.x/nMVDiv+0.5));

                if (IsInFmbd(vb_hit0, hb_hit0, fmbd_blk))

                {

                    int idx_hit0= vb_hit0*nHNum + hb_hit0;

                    MVector_GMV mv_fg= pMV_hit[idx_hit0];

                    if (mv_fg.hitcnt>1)

                    {

                        int mvdiff = (ABS(mv_bg.x - mv_fg.x) + ABS(mv_bg.y - mv_fg.y))/(1<<nMVSubBits);

                        int w_t = MIN(8, MAX(0, mvdiff-2));

                        

                        MVector_GMV *mv_tt = &(pMV_cur[blkidx]);

                        mv_tt->bg_x += w_t*mv_bg.x;  mv_tt->bg_y += w_t*mv_bg.y;  mv_tt->w_fgbg += w_t;

                        mv_tt->fg_x += w_t*mv_fg.x;  mv_tt->fg_y += w_t*mv_fg.y;

                        

                        mv_tt = &(pMV_hit[idx_hit0]);

                        mv_tt->bg_x += w_t*mv_bg.x;  mv_tt->bg_y += w_t*mv_bg.y;  mv_tt->w_fgbg += w_t;

                        mv_tt->fg_x += w_t*mv_fg.x;  mv_tt->fg_y += w_t*mv_fg.y;

                        

                        // hit partner (pMV_hit) with fg,

                        int vb_hit1 = ((int)(vb + nSign*mv_fg.y/nMVDiv+0.5));

                        int hb_hit1 = ((int)(hb + nSign*mv_fg.x/nMVDiv+0.5));

                        int idx_hit1= vb_hit1*nHNum + hb_hit1;

                        if (IsInFmbd(vb_hit1, hb_hit1, fmbd_blk))

                        {   mv_tt = &(pMV_hit[idx_hit1]);

                            if (mv_tt->hitcnt==1)

                            {

                                mv_tt->bg_x += w_t*mv_bg.x;  mv_tt->bg_y += w_t*mv_bg.y;  mv_tt->w_fgbg += w_t;

                                mv_tt->fg_x += w_t*mv_fg.x;  mv_tt->fg_y += w_t*mv_fg.y;

                            }

                        }

                        

                        // hit itself (pMV_cur)

                        int vb_hit2 = ((int)(vb + nSign*(mv_bg.y-mv_fg.y)/nMVDiv+0.5));

                        int hb_hit2 = ((int)(hb + nSign*(mv_bg.x-mv_fg.x)/nMVDiv+0.5));

                        int idx_hit2= vb_hit2*nHNum + hb_hit2;

                        if (IsInFmbd(vb_hit2, hb_hit2, fmbd_blk))

                        {   mv_tt = &(pMV_cur[idx_hit2]);

                            mv_tt->bg_x += w_t*mv_bg.x;  mv_tt->bg_y += w_t*mv_bg.y;  mv_tt->w_fgbg += w_t;

                            mv_tt->fg_x += w_t*mv_fg.x;  mv_tt->fg_y += w_t*mv_fg.y;

                        }

                    }

                }

            }

        }

    }

    

    __int64 glbsum_bgx=0, glbsum_bgy=0, glbsum_fgx=0, glbsum_fgy=0, wfgbg_sum=0;

    for (int blkidx=0; blkidx<nMVSize; blkidx++)

    {

        pMV0[blkidx].fg_x +=  pMV1_hit0[blkidx].fg_x; pMV0[blkidx].fg_y +=  pMV1_hit0[blkidx].fg_y;

        pMV0[blkidx].bg_x +=  pMV1_hit0[blkidx].bg_x; pMV0[blkidx].bg_y +=  pMV1_hit0[blkidx].bg_y;

        pMV0[blkidx].w_fgbg +=  pMV1_hit0[blkidx].w_fgbg;

        

        pMV1[blkidx].fg_x +=  pMV0_hit1[blkidx].fg_x; pMV1[blkidx].fg_y +=  pMV0_hit1[blkidx].fg_y;

        pMV1[blkidx].bg_x +=  pMV0_hit1[blkidx].bg_x; pMV1[blkidx].bg_y +=  pMV0_hit1[blkidx].bg_y;

        pMV1[blkidx].w_fgbg +=  pMV0_hit1[blkidx].w_fgbg;

     

        glbsum_bgx += pMV0[blkidx].bg_x; glbsum_bgy += pMV0[blkidx].bg_y;

        glbsum_bgx += pMV1[blkidx].bg_x; glbsum_bgy += pMV1[blkidx].bg_y;

        

        glbsum_fgx += pMV0[blkidx].fg_x; glbsum_fgy += pMV0[blkidx].fg_y;

        glbsum_fgx += pMV1[blkidx].fg_x; glbsum_fgy += pMV1[blkidx].fg_y;

        

        wfgbg_sum += pMV0[blkidx].w_fgbg;

        wfgbg_sum += pMV1[blkidx].w_fgbg;

    }

    

    wfgbg_sum= MAX(1, wfgbg_sum);

    (*GlbOccMV_o).bg_x = (int)(glbsum_bgx/wfgbg_sum);

    (*GlbOccMV_o).bg_y = (int)(glbsum_bgy/wfgbg_sum);

    

    (*GlbOccMV_o).fg_x = (int)(glbsum_fgx/wfgbg_sum);

    (*GlbOccMV_o).fg_y = (int)(glbsum_fgy/wfgbg_sum);

    

    (*GlbOccMV_o).w_fgbg= (int)MAX(1, (32*wfgbg_sum/MAX(1, nMVSize)));

    

    // for (int blkidx=0; blkidx<nMVSize; blkidx++)

    // {   int w_fgbg = pMV0[blkidx].w_fgbg;

    //     if (w_fgbg>0) { pMV0[blkidx].fg_x = pMV0[blkidx].fg_x/w_fgbg; pMV0[blkidx].fg_y = pMV0[blkidx].fg_y/w_fgbg; pMV0[blkidx].w_fgbg = min(8, w_fgbg); }

    //     if (w_fgbg>0) { pMV0[blkidx].bg_x = pMV0[blkidx].bg_x/w_fgbg; pMV0[blkidx].bg_y = pMV0[blkidx].bg_y/w_fgbg;  }

        

    //     w_fgbg = pMV1[blkidx].w_fgbg;

    //     if (w_fgbg>0) { pMV1[blkidx].fg_x = pMV1[blkidx].fg_x/w_fgbg; pMV1[blkidx].fg_y = pMV1[blkidx].fg_y/w_fgbg; pMV1[blkidx].w_fgbg = min(8, w_fgbg); }

    //     if (w_fgbg>0) { pMV1[blkidx].bg_x = pMV1[blkidx].bg_x/w_fgbg; pMV1[blkidx].bg_y = pMV1[blkidx].bg_y/w_fgbg;  }

    // }

    for (int blkidx=0; blkidx<nMVSize; blkidx++)

        {   int w_fgbg =pMV0[blkidx].w_fgbg>=1?pMV0[blkidx].w_fgbg:1;

            if (w_fgbg>0) { pMV0[blkidx].fg_x = pMV0[blkidx].fg_x/w_fgbg; pMV0[blkidx].fg_y = pMV0[blkidx].fg_y/w_fgbg; pMV0[blkidx].w_fgbg = pMV0[blkidx].w_fgbg<=8?pMV0[blkidx].w_fgbg:8; }

            if (w_fgbg>0) { pMV0[blkidx].bg_x = pMV0[blkidx].bg_x/w_fgbg; pMV0[blkidx].bg_y = pMV0[blkidx].bg_y/w_fgbg;  }

            w_fgbg = pMV1[blkidx].w_fgbg>=1.0?pMV1[blkidx].w_fgbg:1.0;

            if (w_fgbg>0) { pMV1[blkidx].fg_x = pMV1[blkidx].fg_x/w_fgbg; pMV1[blkidx].fg_y = pMV1[blkidx].fg_y/w_fgbg; pMV1[blkidx].w_fgbg =  pMV1[blkidx].w_fgbg<=8? pMV1[blkidx].w_fgbg:8; }

            if (w_fgbg>0) { pMV1[blkidx].bg_x = pMV1[blkidx].bg_x/w_fgbg; pMV1[blkidx].bg_y = pMV1[blkidx].bg_y/w_fgbg;  }

        }

}

    

void VirtualDepthCalc_Core(MVector_GMV *pMV0, MVector_GMV *pMV1, MVector_GlbOccMV *GlbOccMV_i, int nHNum, int nVNum, FmBdInfo fmbd_blk)

{

    int nMVSize = nVNum*nHNum;

    // get virtual depth according to GlbOccMV

    for (int iter=0; iter<2; iter++)

    {

        MVector_GMV *pMV = (iter==0)? pMV0 : pMV1;

        float *pDepth_o = new float[nMVSize];   memset(pDepth_o, 0, nMVSize*sizeof(float));

        float *pDepth_k = new float[nMVSize];   memset(pDepth_k, 0, nMVSize*sizeof(float));

        float *pGain_o  = new float[nMVSize];   memset(pGain_o,  0, nMVSize*sizeof(float));

        float *pGain_k  = new float[nMVSize];   memset(pGain_k,  0, nMVSize*sizeof(float));

        // get depth based on global mv

        for (int blkidx=0; blkidx<nMVSize; blkidx++)

        {

            MVector_GMV mv_t = pMV[blkidx];

            int dist2bg = (ABS(mv_t.x - (*GlbOccMV_i).bg_x) + ABS(mv_t.y - (*GlbOccMV_i).bg_y));//* MIN(8, mv_t.bggain)/8;

            int dist2fg = (ABS(mv_t.x - (*GlbOccMV_i).fg_x) + ABS(mv_t.y - (*GlbOccMV_i).fg_y));//* MIN(8, mv_t.w_fgbg)/8;

            float depth_t = MAX(0, MIN(255, 128 + dist2fg - dist2bg));

            

            pDepth_o[blkidx] = depth_t;

            pGain_o[blkidx]  = (*GlbOccMV_i).w_fgbg;

        }

        int lut_size[5] = {513, 257, 129, 65, 33};

        float lut_wgain[5] = {1, 1, 1, 1, 1};

        

        for (int layer=0; layer<5; layer ++)

        {

            //int layer=3;

            float nGain_DI = lut_wgain[layer];

            int nBlkSize_DI = lut_size[layer];

            Depth_Interpolation(pMV, nBlkSize_DI, nGain_DI, nHNum, nVNum, fmbd_blk, pDepth_k, pGain_k);

            

            for (int blkidx=0; blkidx<nMVSize; blkidx++)

            {

                double w_ki = pGain_k[blkidx];

                double w_pre = pGain_o[blkidx];

                

                w_pre = MAX(0, w_pre - w_ki);

                

                double w_sum = MAX(0.001, w_ki + w_pre);

                

                double w_kk = w_ki/w_sum;

                double w_oo = w_pre/w_sum;

                

                pDepth_o[blkidx] = w_kk*pDepth_k[blkidx] + w_oo*pDepth_o[blkidx];

                pGain_o[blkidx]  = MAX(pGain_k[blkidx], pGain_o[blkidx]);

            }

        }

        

        for (int blkidx=0; blkidx<nMVSize; blkidx++)             pMV[blkidx].depth = 65535*pDepth_o[blkidx]/256;

    

        delete []pDepth_o;  pDepth_o=NULL;

        delete []pGain_o;   pGain_o =NULL;

        delete []pDepth_k;  pDepth_k=NULL;

        delete []pGain_k;   pGain_k =NULL;

    }

}

void Depth_Interpolation(MVector_GMV *pMV_i, int nBlkSize, float w_Gain, int nWidth, int nHeight, FmBdInfo fmbd_i, float *pDepth, float *pGain)

{

    int nHNum = (nWidth + nBlkSize-1)/nBlkSize;

    int nVNum = (nHeight+ nBlkSize-1)/nBlkSize;

    int nMVSize = nVNum*nHNum;

    FmBdInfo fmbd_blk;

    fmbd_blk.StartCol = (fmbd_i.StartCol)/nBlkSize;

    fmbd_blk.StartRow = (fmbd_i.StartRow)/nBlkSize;

    fmbd_blk.EndCol   = (fmbd_i.EndCol + nBlkSize-1)/nBlkSize;

    fmbd_blk.EndRow   = (fmbd_i.EndRow + nBlkSize-1)/nBlkSize;

    

    int nBlkSize_overlap = nBlkSize + 2*(nBlkSize/2);

    int nBlkArea = nBlkSize_overlap*nBlkSize_overlap;

    MVector_GMV *pMV_DS = new MVector_GMV[nMVSize]; memset(pMV_DS, 0, nMVSize*sizeof(MVector_GMV));

    

    // step 1: get block level of fg/bg MV

    int blkidx=0;

    for (int vb=0; vb<nVNum; vb++) for (int hb=0; hb<nHNum; hb++, blkidx++)

    {

        float xsum_bg=0, ysum_bg=0, xsum_fg=0, ysum_fg=0, sum_fggain=0;

        for (int mm=0; mm<nBlkSize_overlap; mm++)  for (int nn=0; nn<nBlkSize_overlap; nn++)

        {

            int row = vb*nBlkSize + mm - (nBlkSize_overlap-nBlkSize)/2;

            int col = hb*nBlkSize + nn - (nBlkSize_overlap-nBlkSize)/2;

            if (IsInFmbd(row, col, fmbd_i))

            {

                MVector_GMV mv_t = pMV_i[row*nWidth + col];

                int w = mv_t.w_fgbg;

                xsum_fg += w*mv_t.fg_x; ysum_fg += w*mv_t.fg_y;

                xsum_bg += w*mv_t.bg_x; ysum_bg += w*mv_t.bg_y;

                

                sum_fggain += mv_t.w_fgbg;  // cnt fg or hole

            }

        }

        sum_fggain = MAX(0.0001, sum_fggain);

        

        MVector_GMV mv_out;

        mv_out.fg_x = xsum_fg/sum_fggain;   mv_out.fg_y = ysum_fg/sum_fggain;

        mv_out.bg_x = xsum_bg/sum_fggain;   mv_out.bg_y = ysum_bg/sum_fggain;

        mv_out.w_fgbg = sum_fggain*32/nBlkArea;

        

        pMV_DS[blkidx] = mv_out;

    }

  

    // setp 2: do bilinear interpolation

    int nB2P_DISTTHR = nBlkSize;

    for (int row =0; row<nHeight; row++)  for (int col=0; col<nWidth; col++)

    {

        int idx = row*nWidth + col;

        MVector_GMV mv_cur = pMV_i[idx];

        int vb=row/nBlkSize; int hb=col/nBlkSize;

        int row_s = row%nBlkSize - nBlkSize/2;

        int col_s = col%nBlkSize - nBlkSize/2;

        

        double depth_sum=0, w_sum=0, wfgbg_sum=0, dist_sum=0;;

        for (int mm=-1; mm<=1; mm++)  for (int nn=-1; nn<=1; nn++)

        {

            int vb_t = MIN(nVNum-1, MAX(0, vb + mm));

            int hb_t = MIN(nHNum-1, MAX(0, hb + nn));

            if (IsInFmbd(vb_t, hb_t, fmbd_blk))

            {

                MVector_GMV mv_ds = pMV_DS[vb_t*nHNum + hb_t];

                

                int w_dist = MAX(0, nB2P_DISTTHR - MAX(ABS(mm*nBlkSize - row_s), ABS(nn*nBlkSize - col_s)));

                int w_cnt = 1 + mv_ds.w_fgbg;

                int w_t = w_dist*w_cnt;

                

                int dist2bg = (ABS(mv_cur.x - mv_ds.bg_x) + ABS(mv_cur.y - mv_ds.bg_y));

                int dist2fg = (ABS(mv_cur.x - mv_ds.fg_x) + ABS(mv_cur.y - mv_ds.fg_y));

                int depth_t = MAX(0, MIN(255, 128 + dist2fg - dist2bg));

                

                depth_sum += w_t*depth_t;

                w_sum += w_t;

                

                wfgbg_sum+= w_dist*mv_ds.w_fgbg;

                dist_sum  += w_dist;

            }

        }

        w_sum = MAX(0.00001, w_sum);

        

        float depth_o = depth_sum/w_sum;

        float wfgbg_o= w_Gain*wfgbg_sum/dist_sum;

        

        pDepth[idx] = depth_o;

        pGain[idx]  = wfgbg_o;

    }

    

    delete []pMV_DS; pMV_DS=NULL;

}

void MvdToFloat(MVD *mv,float* target,int h,int w){

	for (int i = 0; i < w * h; i++)

	{

		target[i*2] = (float)mv[i].depth;

        // target[i*2+1] = (float)mv[i].y;

        // target[i*2] = 1.0;

        // target[i*2+1] = 2.0;

	}

}

extern "C"

void fn(float* mv0,float* mv1,int h,int w)

{

    int nSize_MV = h*w;

    MVD *pP1Phase0MV = new MVD[nSize_MV];  memset(pP1Phase0MV, 0, sizeof(MVD)*nSize_MV);

    MVD *pCFPhase1MV = new MVD[nSize_MV];  memset(pCFPhase1MV, 0, sizeof(MVD)*nSize_MV);

    trans(mv0,pP1Phase0MV,h,w,1);

    trans(mv1,pCFPhase1MV,h,w,1);

    FmBdInfo fmbd;

    fmbd.StartCol =0;

    fmbd.EndCol   =w;

    fmbd.StartRow =0;

    fmbd.EndRow   =h;

    VirtualDepthCalc_Top(pP1Phase0MV,pCFPhase1MV,w,h,fmbd);

    MvdToFloat(pP1Phase0MV,mv0,h,w);

    MvdToFloat(pCFPhase1MV,mv1,h,w);

    delete []pP1Phase0MV; pP1Phase0MV = NULL;

    delete []pCFPhase1MV; pCFPhase1MV = NULL;

    // g++ -o virtualdepth.so -std=c++11 -shared virtual_depth.cpp

    // g++ -o virtualdepth_linux.so -std=c++11 -shared virtual_depth.cpp

}
    // g++ -o virtualdepth_arm64.so -std=c++11 -shared virtual_depth.cpp
