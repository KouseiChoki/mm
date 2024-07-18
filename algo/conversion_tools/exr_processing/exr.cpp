GetDataFromCombEXR(const char *strEXR, FrameData *pFrameData, MVD *pMV0, MVD *pMV1, MVD *pWorldPos)

{

    InputFile inFile0(strEXR);

    

//    bool *bObjMask = new bool[m_nWidth*m_nHeight];

    Imath::Box2i dw = inFile0.header().dataWindow();

    int nWidth = dw.max.x - dw.min.x + 1;

    int nHeight = dw.max.y - dw.min.y + 1;

    m_nInWidth=nWidth;

    m_nInHeight=nHeight;

    m_nOutWidth=nWidth;

    m_nOutHeight=nHeight;

    float *pR,*pG,*pB,*pA;

    pR = new float[nWidth*nHeight*4];

    pG = pR + nWidth*nHeight;//new float[nWidth*nHeight];

    pB = pG + nWidth*nHeight;//new float[nWidth*nHeight];

    pA = pB + nWidth*nHeight;//new float[nWidth*nHeight];

 

    FrameBuffer framebuffer0;

    framebuffer0.insert("A", Slice(FLOAT,(char*)(&pA[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pA[0]) * 1,

                                  sizeof(pA[0]) * nWidth,

                                                1, 1,

                                                0.0));

 

    

    framebuffer0.insert("B", Slice(FLOAT,(char*)(&pB[0] - dw.min.x - dw.min.y * nWidth),

                                      sizeof(pB[0]) * 1,

                                      sizeof(pB[0]) * nWidth,

                                                    1, 1,

                                                    0.0));

 

    framebuffer0.insert("G", Slice(FLOAT,(char*)(&pG[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pG[0]) * 1,

                                  sizeof(pG[0]) * nWidth,

                                                1, 1,

                                                0.0));

 

    framebuffer0.insert("R", Slice(FLOAT,(char*)(&pR[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pR[0]) * 1,

                                  sizeof(pR[0]) * nWidth,

                                                1, 1,

                                                0.0));

    

 

    float *pMVR,*pMVG,*pMVB,*pMVA;

    pMVR = new float[nWidth*nHeight*4];

    pMVG = pMVR+nWidth*nHeight;

    pMVB = pMVG+nWidth*nHeight;

    pMVA = pMVB+nWidth*nHeight;

 

//    FinalImageMovieRenderQueue_MotionVectors.A

    

    framebuffer0.insert("FinalImageMovieRenderQueue_MotionVectors.A", Slice(FLOAT,(char*)(&pMVA[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pA[0]) * 1,

                                  sizeof(pA[0]) * nWidth,

                                                1, 1,

                                                0.0));

 

    

    framebuffer0.insert("FinalImageMovieRenderQueue_MotionVectors.B", Slice(FLOAT,(char*)(&pMVB[0] - dw.min.x - dw.min.y * nWidth),

                                      sizeof(pB[0]) * 1,

                                      sizeof(pB[0]) * nWidth,

                                                    1, 1,

                                                    0.0));

 

    framebuffer0.insert("FinalImageMovieRenderQueue_MotionVectors.G", Slice(FLOAT,(char*)(&pMVG[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pG[0]) * 1,

                                  sizeof(pG[0]) * nWidth,

                                                1, 1,

                                                0.0));

 

    framebuffer0.insert("FinalImageMovieRenderQueue_MotionVectors.R", Slice(FLOAT,(char*)(&pMVR[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pR[0]) * 1,

                                  sizeof(pR[0]) * nWidth,

                                                1, 1,

                                                0.0));

    

    float *pDR,*pDG,*pDB,*pDA;

    pDR = new float[nWidth*nHeight*4];

    pDG = pDR+nWidth*nHeight;

    pDB = pDG+nWidth*nHeight;

    pDA = pDB+nWidth*nHeight;

 

    framebuffer0.insert("FinalImageMovieRenderQueue_WorldDepth.A", Slice(FLOAT,(char*)(&pDA[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pA[0]) * 1,

                                  sizeof(pA[0]) * nWidth,

                                                1, 1,

                                                0.0));

 

    

    framebuffer0.insert("FinalImageMovieRenderQueue_WorldDepth.B", Slice(FLOAT,(char*)(&pDB[0] - dw.min.x - dw.min.y * nWidth),

                                      sizeof(pB[0]) * 1,

                                      sizeof(pB[0]) * nWidth,

                                                    1, 1,

                                                    0.0));

 

    framebuffer0.insert("FinalImageMovieRenderQueue_WorldDepth.G", Slice(FLOAT,(char*)(&pDG[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pG[0]) * 1,

                                  sizeof(pG[0]) * nWidth,

                                                1, 1,

                                                0.0));

 

    framebuffer0.insert("FinalImageMovieRenderQueue_WorldDepth.R", Slice(FLOAT,(char*)(&pDR[0] - dw.min.x - dw.min.y * nWidth),

                                  sizeof(pR[0]) * 1,

                                  sizeof(pR[0]) * nWidth,

                                                1, 1,

                                                0.0));

    

    

    inFile0.setFrameBuffer(framebuffer0);

    inFile0.readPixels(dw.min.y, dw.max.y);

    

//    MVD *pMV         = new MVD[nWidth*nHeight];

//    MVD *pDepth      = new MVD[nWidth*nHeight];

//    FrameData *pData = new FrameData[nWidth*nHeight];

 

    float max_DR = -FLT_MAX,min_DR = FLT_MAX;

    float max_DG = -FLT_MAX,min_DG = FLT_MAX;

    float max_DB = -FLT_MAX,min_DB = FLT_MAX;

//    printf("max float is %f\n",FLT_MAX);

    for(int i=0;i<nWidth*nHeight;i++)

    {

        pMV0[i].x = 2*((pMVR[i] - 0.5) ) * nWidth * -1;

        pMV0[i].y = 2*((pMVG[i] - 0.5) ) * nHeight;

        pMV0[i].z = 0;//pMVB[i];

        pMV0[i].depth = pDR[i];

        pMV0[i].depth = pMVB[i];

        pMV0[i].fb_flag = 0;

        

//        pMV1[i].x = (pMVR[i] - 0.5) * 2;

//        pMV1[i].y = (pMVG[i] - 0.5) * 2;

//        pMV1[i].z = pMVB[i];

//        pMV1[i].depth = pDR[i];

 

        pMV1[i] = pMV0[i];

        

//        pDepth[i].x=pDR[i];

//        pDepth[i].y=pDG[i];

//        pDepth[i].z=pDB[i];

//        pDepth[i].depth=pDA[i];

        

        pWorldPos[i].x     = pDR[i];// scene depth

        pWorldPos[i].y     = pDG[i];// checking

        pWorldPos[i].z     = pDB[i];//absolute depth

        pWorldPos[i].depth = pDA[i];

        

        if(min_DR > pDR[i])

            min_DR = pDR[i];

 

        if(min_DB > pDB[i])

            min_DB = pDB[i];

 

        

        if(max_DR < pDR[i])

            max_DR = pDR[i];

        if(max_DG < pDG[i])

            max_DG = pDG[i];

        if(max_DB < pDB[i])

            max_DB = pDB[i];

 

        pFrameData[i].r=pR[i];

        pFrameData[i].g=pG[i];

        pFrameData[i].b=pB[i];

        pFrameData[i].a=pA[i];

    }

    

    printf("[min max] scene depth is [%f %f], [min max] absolute world depth is [%f %f]\n",min_DR,max_DR,min_DB, max_DB);

//    for(int i=0; i<nWidth*nHeight;i++){

//        pDepth[i].x = pDR[i]/fmax(1,max_DR);

//        pDepth[i].y = pDG[i]/fmax(1,max_DG);

//        pDepth[i].z = pDB[i]/fmax(1,max_DB);

//    }

    

//    Save_Vid_File(pData, nCurIdx);

//    Save_MV_File(pMV, nCurIdx);

//    Save_Depth_File(pDepth, nCurIdx);

    

//    delete []pMV;

//    delete []pDepth;

//    delete []pData;

    delete []pR;     pR     = NULL;

    delete []pMVR;   pMVR   = NULL;

    delete []pDR;    pDR    = NULL;

}