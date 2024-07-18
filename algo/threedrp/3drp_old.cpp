/*
 * @Author: Qing Hong
 * @Date: 2024-01-25 12:44:51
 * @LastEditors: QingHong
 * @LastEditTime: 2024-02-23 11:23:27
 * @Description: file content
 */
#include "stdlib.h"
#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
typedef float DTYPE;
#define D_COL 1579
#define D_ROW 876
#define DD_COL 729
#define DD_ROW 365
#define DEPTH_MAX_VAL 65535
#define TC_EPS 0.00001
#define TC_EPR 0.000001
int m_nZBufferNum;
static int m_nHeight;
static int m_nWidth;
template<typename T> const T& min(const T& a, const T& b) { return a < b ? a : b; }
template<typename T> const T& max(const T& a, const T& b) { return a > b ? a : b; }
class CAlgParms
{
public:
	int m_nFillHoleWithFG = 1;
	int m_nFillHoleWithBGData = 0;
    int m_nMVEdgeThr = 2;
    int m_nMatInputEn = 1;
    int m_nAlphaGainThr = 12;
    int m_nAlphaGainThrOuter = 12;
    int m_nAddEdge = 0;
    int m_nEdgeDetMode = 0;
    int m_nDepth1stEdgeThr =0;
    int m_nObjectRPEn=0;
};

CAlgParms *pstParms = new CAlgParms;

typedef struct _tagEdgeInfo{
    bool u;
    bool l;
}EdgeInfo;

typedef struct _tagPoint2D{
    DTYPE x;
    DTYPE y;
}Point2D;


typedef struct _tagFrameData{
    DTYPE r;
    DTYPE g;
    DTYPE b;
    DTYPE a;
    //for debug
    EdgeInfo e;
}FrameData;

typedef struct _tagMVD{
    DTYPE x;
    DTYPE y;
    DTYPE z;
    DTYPE depth;
        
//    bool fg_pix_edge;
//    bool bg_pix_edge;
//    int  obj_mask;
//    
//    //for Debug
//    int Org_x;
//    int Org_y;
}MVD;

static inline DTYPE DepthTriangleInterp(DTYPE *pTriData,DTYPE *pTriCoef)
{
    DTYPE interp_o = 0;
    DTYPE     *d = pTriData;
    DTYPE alpha = pTriCoef[0];
    DTYPE beta  = pTriCoef[1];
    DTYPE gamma = pTriCoef[2];

    if (alpha<=1 && beta <=1 && alpha>=0 && beta>=0 &&(alpha+beta)<=1 ) {
        interp_o = alpha * d[0] + beta * d[1] + gamma * d[2];
    }else
        interp_o = fmin(fmin(d[0],d[1]),d[2]);
    return interp_o;
}

static inline bool IsInsideTriangle(DTYPE x, DTYPE y, Point2D *pTriVertices)
{
    DTYPE ab_x = pTriVertices[1].x - pTriVertices[0].x, ab_y = pTriVertices[1].y - pTriVertices[0].y;
    DTYPE bc_x = pTriVertices[2].x - pTriVertices[1].x, bc_y = pTriVertices[2].y - pTriVertices[1].y;
    DTYPE ca_x = pTriVertices[0].x - pTriVertices[2].x, ca_y = pTriVertices[0].y - pTriVertices[2].y;

    DTYPE ap_x = x - pTriVertices[0].x,  ap_y = y - pTriVertices[0].y;
    DTYPE bp_x = x - pTriVertices[1].x,  bp_y = y - pTriVertices[1].y;
    DTYPE cp_x = x - pTriVertices[2].x,  cp_y = y - pTriVertices[2].y;

    DTYPE f1=ab_x*ap_y-ab_y*ap_x;
    DTYPE f2=bc_x*bp_y-bc_y*bp_x;
    DTYPE f3=ca_x*cp_y-ca_y*cp_x;
    DTYPE negtive_zero  = 0;//-TC_EPS;
    DTYPE positive_zero = 0;//TC_EPS;

//    if(f1>=negtive_zero&&f2>=negtive_zero&&f3>=negtive_zero)
//        return true;
//    else if(f1<=positive_zero&&f2<=positive_zero&&f3<=positive_zero)
//        return true;
//    else
//        return false;

    if(f1 ==0 && f2 == 0 && f3 == 0)
        return false;
    
    
    if(f1>=negtive_zero&&f2>=negtive_zero&&f3>=negtive_zero)
        return true;
    else
        return false;

    
//    if(f1>negtive_zero&&f2>negtive_zero&&f3>negtive_zero)
//        return true;
//    else if(f1<positive_zero&&f2<positive_zero&&f3<positive_zero)
//        return true;
//    else
//        return false;



}

static inline FrameData FrameDataTriangleInterp(FrameData *pTriData,DTYPE *pTriCoef)
{
    FrameData interp_o;// = 0;
    FrameData     *d = pTriData;
    DTYPE alpha = pTriCoef[0];
    DTYPE beta  = pTriCoef[1];
    DTYPE gamma = pTriCoef[2];

    if (alpha<=1 && beta <=1 && alpha>=0 && beta>=0 &&(alpha+beta)<=1 ) {
        interp_o.r = alpha * d[0].r + beta * d[1].r + gamma * d[2].r;
        interp_o.g = alpha * d[0].g + beta * d[1].g + gamma * d[2].g;
        interp_o.b = alpha * d[0].b + beta * d[1].b + gamma * d[2].b;
        interp_o.a = alpha * d[0].a + beta * d[1].a + gamma * d[2].a;
    }else
    {
        interp_o = d[0] ;//fmin(fmin(d[0],d[1]),d[2]);
    }
    return interp_o;
}

/*
@pTriData: MV in vertices
@pTriCoef: coef of current pixel to three vertices
@return:   interpolated value
*/

static inline MVD MVTriangleInterp(MVD *pTriData,DTYPE *pTriCoef)
{
    MVD interp_o;// = 0;
    MVD      *d = pTriData;
    DTYPE alpha = pTriCoef[0];
    DTYPE beta  = pTriCoef[1];
    DTYPE gamma = pTriCoef[2];

    if (alpha<=1 && beta <=1 && alpha>=0 && beta>=0 &&(alpha+beta)<=1 ) {
        interp_o.x = alpha * d[0].x + beta * d[1].x + gamma * d[2].x;
        interp_o.y = alpha * d[0].y + beta * d[1].y + gamma * d[2].y;
        interp_o.z = alpha * d[0].z + beta * d[1].z + gamma * d[2].z;
    }else
    {
        interp_o = d[0] ;//fmin(fmin(d[0],d[1]),d[2]);
    }
    
//    interp_o.fg_pix_edge = d[0].fg_pix_edge || d[1].fg_pix_edge || d[2].fg_pix_edge ;
//    interp_o.bg_pix_edge = d[0].bg_pix_edge || d[1].bg_pix_edge || d[2].bg_pix_edge ;

//    interp_o = d[0];
    return interp_o;
}


static inline void ComputeBarycentric2D(DTYPE x, DTYPE y, Point2D *pTriVertices, DTYPE *pTriCoef_O)
{
        Point2D v1 = pTriVertices[0];
        Point2D v2 = pTriVertices[1];
        Point2D v3 = pTriVertices[2];
        //
        DTYPE detT   = (v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y);
        DTYPE lamda1 = (v2.y - v3.y) * (x    - v3.x) + (v3.x - v2.x) * (y - v3.y );
        DTYPE lamda2 = (v3.y - v1.y) * (x    - v3.x) + (v1.x - v3.x) * (y - v3.y );
        
        if (fabs(detT) <= 0.000001) {
            lamda1 = 0;
            lamda2 = 0;
        }else{
            lamda1 = lamda1/ detT;
            lamda2 = lamda2/ detT;
        }
    
        DTYPE lamda3 = 1 - lamda1 - lamda2;    
        //    original algorithm, not stable,deprecated, the formula is from [Fundamentals of Computer Graphics (4th Edition)] page62
        //    DTYPE alpha = (x*(v[1].y - v[2].y) + (v[2].x - v[1].x)*y + v[1].x*v[2].y - v[2].x*v[1].y) / (v[0].x*(v[1].y - v[2].y) + (v[2].x - v[1].x)*v[0].y + v[1].x*v[2].y - v[2].x*v[1].y);
        //    DTYPE beta  = (x*(v[2].y - v[0].y) + (v[0].x - v[2].x)*y + v[2].x*v[0].y - v[0].x*v[2].y) / (v[1].x*(v[2].y - v[0].y) + (v[0].x - v[2].x)*v[1].y + v[2].x*v[0].y - v[0].x*v[2].y);
        //
        //    alpha = fmax(0, fmin(1,alpha));
        //    beta = fmax(0, fmin(1,beta));
        //    gamma = fmax(0, fmin(1,gamma));
            
        //    DTYPE c3    = (x*(v[0].y - v[1].y) + (v[1].x - v[0].x)*y + v[0].x*v[1].y - v[1].x*v[0].y) / (v[2].x*(v[0].y - v[1].y) + (v[1].x - v[0].x)*v[2].y + v[0].x*v[1].y - v[1].x*v[0].y);
        //    DTYPE gamma = 1 - alpha - beta;

        //    alpha = (x*(v[1].y - v[2].y) + (v[2].x - v[1].x)*y + v[1].x*v[2].y - v[2].x*v[1].y) / (v[0].x*(v[1].y - v[2].y) + (v[2].x - v[1].x)*v[0].y + v[1].x*v[2].y - v[2].x*v[1].y);
        //    beta  = (x*(v[2].y - v[0].y) + (v[0].x - v[2].x)*y + v[2].x*v[0].y - v[0].x*v[2].y) / (v[1].x*(v[2].y - v[0].y) + (v[0].x - v[2].x)*v[1].y + v[2].x*v[0].y - v[0].x*v[2].y);
        //    DTYPE c3    = (x*(v[0].y - v[1].y) + (v[1].x - v[0].x)*y + v[0].x*v[1].y - v[1].x*v[0].y) / (v[2].x*(v[0].y - v[1].y) + (v[1].x - v[0].x)*v[2].y + v[0].x*v[1].y - v[1].x*v[0].y);
        //    gamma = 1 - alpha - beta;

        pTriCoef_O[0] = lamda1;
        pTriCoef_O[1] = lamda2;
        pTriCoef_O[2] = lamda3;
}


void trans(float* data,MVD* result,int nHeight,int nWidth){

     for(int nRow = 0; nRow < nHeight; nRow++)

        {

            for(int nCol = 0; nCol < nWidth; nCol++)

            {

                int nIndex = nCol + nRow * nWidth;

                float xx = (float)data[nIndex*4];

                float yy = (float)data[nIndex*4+1];

                float zz = (float)data[nIndex*4+2];

                float aa = (float)data[nIndex*4+3];

                result[nIndex].x = xx;

                result[nIndex].y = yy;

                result[nIndex].z = zz;

                result[nIndex].depth = aa;

        }

}
}

void MotionEdgeDet(MVD *pMVIn, EdgeInfo *pEdgeO)
{
    DTYPE nMVEdgeThr = pstParms->m_nMVEdgeThr;
    int nWidth  = m_nWidth;
    int nHeight = m_nHeight;
    
    DTYPE x_gain = nWidth;
    DTYPE y_gain = nHeight;
    DTYPE z_gain = 0;
    DTYPE FG_BG_Depth_Thr = 1;//pstParms->m_BG_FG_DEPTH_THR;
    for(int y=0; y<nHeight; y++){
            for(int x=0; x<nWidth;  x++){
                
                if(x == D_COL && y == D_ROW)
                    int dd = 0;
                if(x == 676 && y == 0)
                    int dd = 0;
                
                int nIndex = x + y * nWidth;
                
                
                int u[4]={x, x+1, x, x+1};
                int v[4]={y, y, y+1, y+1};
                MVD mv_[4] = {0};
                DTYPE depth_[4] = {0};
                for(int idx=0;idx<4;idx++)
                {
                    u[idx] = std::max(0, std::min(u[idx], nWidth-1 ));
                    v[idx] = std::max(0, std::min(v[idx], nHeight-1));
                    
                    int cx=u[idx];
                    int cy=v[idx];
                    int cidx=cy*nWidth+cx;
                    mv_[idx] = pMVIn[cidx];
                    depth_[idx] = mv_[idx].depth;
                    mv_[idx].x /= x_gain;
                    mv_[idx].y /= y_gain;
                }
                
                
                DTYPE maxMVX = 0, maxMVY = 0, maxMV = 0;
                maxMVX = max(max(fabs(mv_[0].x), fabs(mv_[1].x)),max(fabs(mv_[2].x), fabs(mv_[3].x)));
                maxMVY = max(max(fabs(mv_[0].y), fabs(mv_[1].y)),max(fabs(mv_[2].y), fabs(mv_[3].y)));
                maxMV = max(maxMVX, maxMVY);
//                nMVEdgeThr = (maxMV*pstParms->m_nMVEdgeGain/16 + (DTYPE)pstParms->m_nMVEdgeThr);//nMVEdgeThr;
                DTYPE edge_l, edge_u;
                
                edge_u=fmax(fabs(mv_[0].x-mv_[1].x)*x_gain +
                            fabs(mv_[0].y-mv_[1].y)*y_gain,
                            
                            fabs(mv_[0].x-mv_[2].x)*x_gain +
                            fabs(mv_[0].y-mv_[2].y)*y_gain );//+fabs(mv_[0].z-mv_[2].z)*z_gain);
                
                edge_u=fmax(edge_u,fabs(mv_[2].x-mv_[1].x) * x_gain +
                            fabs(mv_[2].y-mv_[1].y) * y_gain );//+
                                //fabs(mv_[2].z-mv_[1].z) * z_gain);
                
                edge_l=fmax(fabs(mv_[3].x-mv_[1].x) * x_gain  +
                            fabs(mv_[3].y-mv_[1].y) * y_gain ,
                            fabs(mv_[3].x-mv_[2].x) * x_gain  +
                            fabs(mv_[3].y-mv_[2].y) * y_gain  );
                edge_l=fmax(edge_l, fabs(mv_[2].x-mv_[1].x) * x_gain +
                                    fabs(mv_[2].y-mv_[1].y) * y_gain);
                
                    if (edge_u > nMVEdgeThr) {
                        pEdgeO[nIndex].u = true;
                    }else
                        pEdgeO[nIndex].u = false;
                    
                    if (edge_l > nMVEdgeThr) {
                        pEdgeO[nIndex].l = true;
                    }else
                        pEdgeO[nIndex].l = false;
                

                
        }
    }
}

//Detect Edge with Alpha Mask, for 2D to 3D
void AlphaMatteEdgeDet(FrameData *pInData, EdgeInfo *pEdgeO)
{
    int imSize = m_nWidth * m_nHeight;
    
    int nWidth  = m_nWidth;
    int nHeight = m_nHeight;
    
    DTYPE alphaThr = 0.0;
    
    for(int y=0; y<nHeight; y++){
            for(int x=0; x<nWidth;  x++){
                int nIndex = x + y * nWidth;
                int u[4]={x, x+1, x, x+1};
                int v[4]={y, y, y+1, y+1};
                DTYPE alpha_data_[4] = {0};
                for(int idx=0;idx<4;idx++)
                {
                    u[idx] = std::max(0, std::min(u[idx], nWidth-1 ));
                    v[idx] = std::max(0, std::min(v[idx], nHeight-1));
                    
                    int cx=u[idx];
                    int cy=v[idx];
                    int cidx=cy*nWidth+cx;
                    alpha_data_[idx] = pInData[cidx].a;
                }
                
                if(x == D_COL && y == D_ROW)
                    int d = 0;
                
                if(alpha_data_[0]>alphaThr && (alpha_data_[1] <= alphaThr || alpha_data_[2] <= alphaThr))
                    pEdgeO[nIndex].u = true;
                else
                    pEdgeO[nIndex].u = false;
                
                if((alpha_data_[1] > alphaThr && alpha_data_[2] > alphaThr && alpha_data_[3] > alphaThr ) ||
                   (alpha_data_[1] <= alphaThr && alpha_data_[2] <= alphaThr && alpha_data_[3] <= alphaThr ))
                    pEdgeO[nIndex].l = false;
                else
                    pEdgeO[nIndex].l = true;
                
        }
    }
    
}


void MCRender(FrameData *pTriDataIn_, MVD *pTriMVFI_, Point2D *pRenderTriVeterx_, DTYPE *pTriDepthFI_, DTYPE x, DTYPE y, DTYPE *pZBuffer, FrameData *pDataOut, int offset, int EdgeFlag, MVD *pMVOut )
{
    Point2D       *pRenderTriVeterx = &pRenderTriVeterx_[offset];
    FrameData     *pTriDataIn       = &pTriDataIn_[offset];
    DTYPE         *pTriDepthFI      = &pTriDepthFI_[offset];
    MVD           *pTriMV           = &pTriMVFI_[offset];
    int cdd = 0;
    DTYPE x_min = fmin(pRenderTriVeterx[0].x, pRenderTriVeterx[1].x);
    DTYPE x_max = fmax(pRenderTriVeterx[0].x, pRenderTriVeterx[1].x);
    
    DTYPE y_min = fmin(pRenderTriVeterx[0].y, pRenderTriVeterx[1].y);
    DTYPE y_max = fmax(pRenderTriVeterx[0].y, pRenderTriVeterx[1].y);
        
    x_min = fmin(x_min, pRenderTriVeterx[2].x);
    x_max = fmax(x_max, pRenderTriVeterx[2].x);
    y_min = fmin(y_min, pRenderTriVeterx[2].y);
    y_max = fmax(y_max, pRenderTriVeterx[2].y);
    
    if (y == D_ROW && x == D_COL) {
        int c = 0;
    }
    
    DTYPE pix_center = 0;//0.5;
    DTYPE off_x , off_y;
    off_x = pix_center;
    off_y = pix_center;
    int x_min_int = (int)floor(x_min );//+ off_x);
    int y_min_int = (int)floor(y_min );//+ off_y);
    int isXInt = fabs(x_min - x_min_int) < TC_EPS ? 1:0;
    int isYInt = fabs(y_min - y_min_int) < TC_EPS ? 1:0;
    int x_max_int = (int)floor(x_max );//+ off_x);
    int y_max_int = (int)floor(y_max );//+ off_y);
        
    int upleft_x = x_min_int;
    int upleft_y = y_min_int;
    int x_num = 1;
    int y_num = 1;
    /* original logic
    if (isXInt)
    {
        x_num=(int) x_max- (int)x_min+1;
        upleft_x = (int)x_min;
    }
    else
    {
        x_num=(int) x_max- (int)x_min;
        upleft_x = (int)x_min + 1;
    }
    
    if (isYInt)
    {
        y_num=(int) y_max- (int)y_min+1;
        upleft_y = (int)y_min;
    }
    else
    {
        y_num=(int) y_max- (int)y_min;
        upleft_y = (int) y_min + 1;
    }
     */
    
    //the logic is changed to process the first row/column issue
    if (isXInt)
    {
        x_num= x_max_int - x_min_int + 1;//(int) x_max- (int)x_min+1;
        upleft_x = x_min_int;// (int)x_min;
    }
    else
    {
        x_num= x_max_int - x_min_int;//(int) x_max- (int)x_min;
        upleft_x =  x_min_int + 1;//(int)x_min + 1;
    }
    
    if (isYInt)
    {
        y_num=  y_max_int - y_min_int + 1  ;//(int) y_max- (int)y_min+1;
        upleft_y =   y_min_int;//(int)y_min;
    }
    else
    {
        y_num= y_max_int - y_min_int ;//(int) y_max- (int)y_min;
        upleft_y = y_min_int + 1 ;//(int) y_min + 1;
    }

    
    int nHitNum = 0;
    
    if (x_num*y_num>10) {
        int dd = 0;
    }
    
    for (int jj = 0 ; jj<y_num; jj++)
        for (int ii = 0; ii<x_num; ii++)
        {
            int x_out = upleft_x + ii;
            int y_out = upleft_y + jj;
            cdd += 1;
            x_out = std::max(0,std::min(m_nWidth-1,x_out));
            y_out = std::max(0,std::min(m_nHeight-1, y_out));
            
            if (y_out == D_ROW && x_out == D_COL) {
                
                // printf("[up x, up y] = [%d %d]\n", upleft_x, upleft_y);
                int c = 0;
            }
            
            if(x_out == 1 || y_out == 1){
                int dd = 0;
            }
            
            int nIndexOut = x_out + y_out * m_nWidth;
            
            if (IsInsideTriangle((DTYPE)x_out + off_x, (DTYPE)y_out + off_y, pRenderTriVeterx))
            {
                nHitNum++;
                
                if (y_out == D_ROW && x_out == D_COL) {
                    int c = 0;
                    m_nZBufferNum++;
                    // printf("[nRow nCol] = [%d %d], Z buffer number is %d\n",y_out,x_out,m_nZBufferNum);
                }
                
                DTYPE pTriCoef[3];
                ComputeBarycentric2D((DTYPE)x_out + off_x, (DTYPE)y_out + off_y, pRenderTriVeterx,  pTriCoef);
                DTYPE nInterpDepth = DepthTriangleInterp(pTriDepthFI, pTriCoef);
                if (nInterpDepth < pZBuffer[nIndexOut] && EdgeFlag == 0)
                {
                    pZBuffer[nIndexOut]       = nInterpDepth;
                    pDataOut[nIndexOut]       = FrameDataTriangleInterp(pTriDataIn, pTriCoef);
                    pMVOut[nIndexOut]         = MVTriangleInterp(pTriMV, pTriCoef);
                    pMVOut[nIndexOut].depth   = nInterpDepth ;//MVTriangleInterp(pTriMV, pTriCoef);
                }
            }
        }
}


void FI3D_Core(MVD *pMVIn, FrameData *pDataIn, DTYPE *pZBufferIn, DTYPE nPhaseIn ,FrameData *pFIBuffer,MVD *pFIMVOut)
{
    // printf("line %d ,pDataIn in [0 0] is [%d] = [%f]\n", __LINE__ ,pDataIn[0]);
    int nHeight = m_nHeight;
    int nWidth  = m_nWidth;
    m_nZBufferNum = 0;
    //initial depth as an infinite value
    for (int ii = 0; ii<nHeight * nWidth; ii++) {
        pZBufferIn[ii] = DEPTH_MAX_VAL;//__FLT_MAX__;
        pFIMVOut[ii].depth = DEPTH_MAX_VAL;
    }
    

    int nDepthThr = pstParms->m_nDepth1stEdgeThr;
    
    EdgeInfo *pMotEdge =  new EdgeInfo[nWidth*nHeight];
    if (pstParms->m_nEdgeDetMode==0) {
        MotionEdgeDet(pMVIn, pMotEdge);
    }else  
    {
        AlphaMatteEdgeDet(pDataIn, pMotEdge);
    }
//    else if(pstParms->m_nEdgeDetMode>3)
//        MixedEdgeDet(pMVIn, pMotEdge);
        
//    DTYPE nMVEdgeThr = pstParms->m_nMVEdgeThr;
//    constexpr int SUBPIXEL_PRECISION_BITS = 0 ? 4 : 8;
//    constexpr float SUBPIXEL_PRECISION_FACTOR = static_cast<float>(1 << SUBPIXEL_PRECISION_BITS);
//    constexpr int SUBPIXEL_PRECISION_MASK = 0xFFFFFFFF >> (32 - SUBPIXEL_PRECISION_BITS);
    int cdd = 0;
    for(int y=0; y<nHeight; y++) {
        for(int x=0; x<nWidth;  x++){
                        
            int u[4]    =   {x, x+1, x, x+1};
            int v[4]    =   {y, y, y+1, y+1};
            
            int nIndex = x + y * nWidth;
                        
            DTYPE depth0=100000;
            DTYPE depth1=100000;
            
            int depth0_num=0;
            int depth1_num=3;
            int depth0_num_max=0;
            int depth1_num_max=3;
            int depth0_max = 0;
            int depth1_max = 0;
            
            
            Point2D       pInTriVertex[4];
            Point2D       pOutTriVertex[4];
            DTYPE         nDepthFI[4];
            DTYPE         nDepthFI_tmp[4];
            FrameData     pInTriData[4];
            FrameData     pInTriDataTmp[4];
            DTYPE         alphaData[4];
            MVD mv_[4]   = {0};
            MVD mv_fi[4] = {0};
            MVD mv_fi_tmp[4] = {0};
            for(int idx=0;idx<4;idx++)
            {
                u[idx] = std::max(0, std::min(u[idx], nWidth-1 ));
                v[idx] = std::max(0, std::min(v[idx], nHeight-1));
                
                int cx=u[idx];
                int cy=v[idx];                
                
                int cidx=cy*nWidth+cx;
                pInTriData[idx] = pDataIn[cidx];
                pInTriDataTmp[idx] = pInTriData[idx];
                     
                
                pInTriVertex[idx].x=cx;
                pInTriVertex[idx].y=cy;
                alphaData[idx] = pInTriData[idx].a;
                //pixel center is [0 0] or [0.5 0.5]
                float off_x = 0;//0.5;
                float off_y = 0;//0.5;
                pOutTriVertex[idx].x=pMVIn[cidx].x * nPhaseIn  + pInTriVertex[idx].x + off_x;
                pOutTriVertex[idx].y=pMVIn[cidx].y * nPhaseIn  + pInTriVertex[idx].y + off_y;
                                
                mv_fi[idx].x = pMVIn[cidx].x * nPhaseIn;
                mv_fi[idx].y = pMVIn[cidx].y * nPhaseIn;
                mv_fi[idx].z = pMVIn[cidx].z * nPhaseIn;
                mv_fi_tmp[idx] = mv_fi[idx];
                mv_[idx] = pMVIn[cidx];
                
                nDepthFI[idx] = mv_[idx].depth + mv_[idx].z * nPhaseIn;
                nDepthFI[idx] = fmin(nDepthFI[idx], DEPTH_MAX_VAL);
                nDepthFI_tmp[idx] = nDepthFI[idx];
                
//                __FLT_MAX__;
//                __FLT_MIN__;

                
                if (idx<3 && mv_[idx].depth<depth0){
                    depth0=mv_[idx].depth;
                    depth0_num=idx;
                }
                
                if (idx<3 && mv_[idx].depth>depth0_max){
                    depth0_max=mv_[idx].depth;
                    depth0_num_max=idx;
                }
                
                if (idx>0 && mv_[idx].depth<depth1){
                    depth1= mv_[idx].depth;
                    depth1_num=idx;
                }
                
                if (idx>0 && mv_[idx].depth>depth1_max){
                    depth1_max= mv_[idx].depth;
                    depth1_num_max=idx;
                }
            }
            
            bool isFIEnable_u = 1 ;//- (edge_u>nMVEdgeThr);//pMotEdge[nIndex].u;
            bool isFIEnable_l = 1 ;//- (edge_l>nMVEdgeThr);//pMotEdge[nIndex].l;
            
            int  nAddEdge_u = pstParms->m_nAddEdge;
            int  nAddEdge_l = pstParms->m_nAddEdge;
            if(x == 676 && y == 0)
                int dd = 0;
            isFIEnable_u = 1 - pMotEdge[nIndex].u;
            isFIEnable_l = 1 - pMotEdge[nIndex].l;
            
            EdgeInfo e;
            e.u = 1 - isFIEnable_u;
            e.l = 1 - isFIEnable_l;
            
            
            pDataIn[nIndex].e = e;//(isFIEnable_u || isFIEnable_l) ;//pMotEdge[nIndex];
            
            Point2D pOutTriVerextBuff[4];
            pOutTriVerextBuff[0] = pOutTriVertex[0];
            pOutTriVerextBuff[1] = pOutTriVertex[1];
            pOutTriVerextBuff[2] = pOutTriVertex[2];
            pOutTriVerextBuff[3] = pOutTriVertex[3];
            
            if (pstParms->m_nAddEdge > 0 && isFIEnable_u == 0 ) {
                for (int i=0; i<3; i++) {
                                        
                    pOutTriVertex[i].x=mv_[depth0_num].x * nPhaseIn  + pInTriVertex[i].x ;//+ off_x;
                    pOutTriVertex[i].y=mv_[depth0_num].y * nPhaseIn  + pInTriVertex[i].y ;//+ off_y;
                    
                    mv_fi[i].x = mv_[depth0_num].x * nPhaseIn;
                    mv_fi[i].y = mv_[depth0_num].y * nPhaseIn;
                    mv_fi[i].z = mv_[depth0_num].z * nPhaseIn;
                    if (pstParms->m_nAddEdge==2) {//use FG data to interpolate
                        pInTriData[i] = pInTriDataTmp[depth0_num];
                    }

                    if(pstParms->m_nAddEdge==3 && fabs(mv_[0].depth - mv_[depth0_num].depth)<nDepthThr){// if current pixel is FG data, then add edge; if current pixel is BG data, no projection
                        pInTriData[i] = pInTriDataTmp[depth0_num];
                    }
                    nDepthFI[i] = mv_[depth0_num].depth + mv_[depth0_num].z * nPhaseIn;
                    nDepthFI[i] = fmin(nDepthFI[i], DEPTH_MAX_VAL);
                }
            }
            
            if (pstParms->m_nAddEdge==2) {//use FG data to interpolate, Testing, to fix the un-expected color in object edge
                pInTriData[0] = pInTriDataTmp[0];
                pInTriData[1] = pInTriDataTmp[1];
                pInTriData[2] = pInTriDataTmp[2];
                pInTriData[3] = pInTriDataTmp[3];
            }
            
            bool has_fb = 0;
            bool up_alpha_en  = 1;
            if(pstParms->m_nObjectRPEn)
                up_alpha_en = alphaData[0]>0 &&alphaData[1]>0 &&alphaData[2]>0 ;
            
            isFIEnable_u = isFIEnable_u && up_alpha_en;
            
            nDepthFI[0] = fmax(0, fmin(nDepthFI[0], DEPTH_MAX_VAL - 1));
            nDepthFI[1] = fmax(0, fmin(nDepthFI[1], DEPTH_MAX_VAL - 1 ));
            nDepthFI[2] = fmax(0, fmin(nDepthFI[2], DEPTH_MAX_VAL - 1 ));
            
            //**********
            //  0    1
            //  2    3
            // 1-2-3: counter-clock-wise(ccw)
            // 1-3-2: clock-wise(cw)
            // ccw or cw influences the judgement rule of whether a pixel is inside a triangle
            
            
            if(y == DD_ROW && x == DD_COL)
            {
                // printf("line %d ,Vertex in [R C] is [%d %d] = [%f %f]\n", __LINE__ ,y , x ,pOutTriVertex[0].y, pOutTriVertex[0].x);
                int d1 = 0;
            }

            
            if ((isFIEnable_u || (nAddEdge_u>0 && isFIEnable_u==0)) && has_fb==0) {
                MCRender(pInTriData, mv_fi,pOutTriVertex, nDepthFI, x, y, pZBufferIn, pFIBuffer,0,0,pFIMVOut);
            }
            else
            {
                if (pstParms->m_nFillHoleWithFG) {
                    for (int idx = 0; idx < 3; idx++) {
                        pOutTriVertex[idx].x=mv_[idx].x * nPhaseIn  + pInTriVertex[idx].x ;//FGMV
                        pOutTriVertex[idx].y=mv_[idx].y * nPhaseIn  + pInTriVertex[idx].y ;//FGMV;
                        mv_fi[idx].x = mv_[depth0_num].x * nPhaseIn;
                        mv_fi[idx].y = mv_[depth0_num].y * nPhaseIn;
                        mv_fi[idx].z = mv_[depth0_num].z * nPhaseIn;
                        nDepthFI[idx] = mv_[depth0_num].depth + mv_[depth0_num].z * nPhaseIn; //BGData
                        nDepthFI[idx] = fmin(nDepthFI[idx], DEPTH_MAX_VAL);
                        pInTriData[idx] = pInTriData[depth0_num];//bgdata
                    }
                    MCRender(pInTriData, mv_fi,pOutTriVertex, nDepthFI, x, y, pZBufferIn, pFIBuffer,0,1,pFIMVOut);
                }
            }
 
            pOutTriVertex[0] = pOutTriVerextBuff[0];
            pOutTriVertex[1] = pOutTriVerextBuff[1];
            pOutTriVertex[2] = pOutTriVerextBuff[2];
            pOutTriVertex[3] = pOutTriVerextBuff[3];
            nDepthFI[0] = nDepthFI_tmp[0];
            nDepthFI[1] = nDepthFI_tmp[1];
            nDepthFI[2] = nDepthFI_tmp[2];
            nDepthFI[3] = nDepthFI_tmp[3];
            mv_fi[0] = mv_fi_tmp[0];
            mv_fi[1] = mv_fi_tmp[1];
            mv_fi[2] = mv_fi_tmp[2];
            mv_fi[3] = mv_fi_tmp[3];

            
            pInTriData[0] = pInTriDataTmp[0];
            pInTriData[1] = pInTriDataTmp[1];
            pInTriData[2] = pInTriDataTmp[3];
            pInTriData[3] = pInTriDataTmp[2];
            pOutTriVertex[0] = pOutTriVerextBuff[0];
            pOutTriVertex[1] = pOutTriVerextBuff[1];
            pOutTriVertex[2] = pOutTriVerextBuff[3];
            pOutTriVertex[3] = pOutTriVerextBuff[2];
            nDepthFI[0] = nDepthFI_tmp[0];
            nDepthFI[1] = nDepthFI_tmp[1];
            nDepthFI[2] = nDepthFI_tmp[3];
            nDepthFI[3] = nDepthFI_tmp[2];
            mv_fi[0] = mv_fi_tmp[0];
            mv_fi[1] = mv_fi_tmp[1];
            mv_fi[2] = mv_fi_tmp[3];
            mv_fi[3] = mv_fi_tmp[2];

            
            nDepthFI[0] = fmax(0, fmin(nDepthFI[0], DEPTH_MAX_VAL - 1));
            nDepthFI[1] = fmax(0, fmin(nDepthFI[1], DEPTH_MAX_VAL - 1));
            nDepthFI[2] = fmax(0, fmin(nDepthFI[2], DEPTH_MAX_VAL - 1));
            nDepthFI[3] = fmax(0, fmin(nDepthFI[3], DEPTH_MAX_VAL - 1));
            if (y==0 && x ==541){
                int a = 0;
            }
            if (pstParms->m_nAddEdge>0 && isFIEnable_l == 0) {
                for (int i=1; i<4; i++) {
                    
                    pOutTriVertex[i].x=mv_[depth1_num].x * nPhaseIn  + pInTriVertex[i].x ;//+ off_x;
                    pOutTriVertex[i].y=mv_[depth1_num].y * nPhaseIn  + pInTriVertex[i].y ;//+ off_y;
                    
                    mv_fi[i].x = mv_[depth1_num].x * nPhaseIn;
                    mv_fi[i].y = mv_[depth1_num].y * nPhaseIn;
                    mv_fi[i].z = mv_[depth1_num].z * nPhaseIn;
                    if (pstParms->m_nAddEdge==2) {//use FG data to interpolate
                        pInTriData[i] = pInTriDataTmp[depth1_num];
                    }
                    
                    if(pstParms->m_nAddEdge==3 && fabs(mv_[3].depth - mv_[depth1_num].depth)<nDepthThr){// if current pixel is FG data, then add edge; if current pixel is BG data, no projection
                        pInTriData[i] = pInTriDataTmp[depth1_num];
                    }

                    nDepthFI[i] = mv_[depth1_num].depth + mv_[depth1_num].z * nPhaseIn;
                    nDepthFI[i] = fmin(nDepthFI[i], DEPTH_MAX_VAL);
                }
            }

            has_fb = 0;
            bool low_alpha_en  = 1;
            if(pstParms->m_nObjectRPEn)
                low_alpha_en = alphaData[3]>0 &&alphaData[1]>0 &&alphaData[2]>0 ;
            isFIEnable_l = isFIEnable_l && low_alpha_en;
            
            if ((isFIEnable_l || (nAddEdge_l>0&& isFIEnable_l == 0)) && has_fb==0) {
                MCRender(pInTriData, mv_fi, pOutTriVertex, nDepthFI, x, y, pZBufferIn, pFIBuffer,1,0,pFIMVOut);

            }else
            {
                if (pstParms->m_nFillHoleWithFG) {
                    for (int idx = 1; idx < 4; idx++) {
                        pOutTriVertex[idx].x=mv_[idx].x * nPhaseIn  + pInTriVertex[idx].x ;//+ off_x;
                        pOutTriVertex[idx].y=mv_[idx].y * nPhaseIn  + pInTriVertex[idx].y ;//+ off_y;
                        nDepthFI[idx] = mv_[depth1_num].depth + mv_[depth1_num].z * nPhaseIn;
                        nDepthFI[idx] = fmin(nDepthFI[idx], DEPTH_MAX_VAL);
                        pInTriData[idx] = pInTriData[depth1_num];
                        mv_fi[idx].x = mv_[depth1_num].x * nPhaseIn;
                        mv_fi[idx].y = mv_[depth1_num].y * nPhaseIn;
                        mv_fi[idx].z = mv_[depth1_num].z * nPhaseIn;
                    }
                    MCRender(pInTriData, mv_fi,pOutTriVertex, nDepthFI, x, y, pZBufferIn, pFIBuffer,1,1,pFIMVOut);
                }
            }
        }
        
    }
    
    // printf("finalcdd=%d\n",cdd);
    
    delete []pMotEdge;
}

void trans_img(float* data,FrameData* result,int nHeight,int nWidth){

     for(int nRow = 0; nRow < nHeight; nRow++)

        {

            for(int nCol = 0; nCol < nWidth; nCol++)

            {

                int nIndex = nCol + nRow * nWidth;

                int xx = (float)data[nIndex*4];

                int yy = (float)data[nIndex*4+1];

                int zz = (float)data[nIndex*4+2];

                int aa = (float)data[nIndex*4+3];

                result[nIndex].r = xx;

                result[nIndex].g = yy;

                result[nIndex].b = zz;

                result[nIndex].a = aa;

        }

}
}

void MvdToFloat(FrameData *image,float* target,int h,int w){

	for (int i = 0; i < w * h; i++)

	{

		target[i*4] = (float)image[i].r;
        target[i*4+1] = (float)image[i].g;
        target[i*4+2] = (float)image[i].b;
        target[i*4+3] = (float)image[i].a;

	}

}

extern "C"

void fn(float* image,float* mv,int h,int w)

{
    m_nHeight = h;
    m_nWidth = w;

    int nImSize = h*w;

    MVD *pMVIn = new MVD[nImSize];  memset(pMVIn, 0, sizeof(MVD)*nImSize);

    FrameData *pDataIn = new FrameData[nImSize];  memset(pMVIn, 0, sizeof(FrameData)*nImSize);
    trans(mv,pMVIn,h,w);
    trans_img(image,pDataIn,h,w);

    FrameData *pRPOut = new FrameData[nImSize];
    MVD       *pRPMVD  = new MVD[nImSize];
    DTYPE     *pDepthO = new DTYPE[nImSize];    
    memset(pRPOut,  0, sizeof(FrameData)*nImSize);
    memset(pRPMVD,  0, sizeof(MVD)*nImSize);
    memset(pDepthO, 0, sizeof(DTYPE)*nImSize);

    FI3D_Core(pMVIn, pDataIn, pDepthO, 1, pRPOut, pRPMVD);

    MvdToFloat(pRPOut,image,h,w);

    delete []pMVIn; pMVIn = NULL;

    delete []pDataIn; pDataIn = NULL;

    delete []pRPOut; pRPOut = NULL;

    delete []pRPMVD; pRPMVD = NULL;

    delete []pDepthO; pDepthO = NULL;


    // g++ -o 3drp_arm64.so -std=c++11 -shared 3drp.cpp
    // g++ -o 3drp.so -std=c++11 -shared 3drp.cpp
}