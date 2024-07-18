import numpy as np
Slope = 0.98
Toe = 0.3
Shoulder = 0.22
BlackClip = 0
WhiteClip = 0.025
import imageio
import colour
linearCV = imageio.imread('/Users/qhong/Desktop/1.0000.exr')[...,:3]
XYZ_2_AP0_MAT = np.array(
	[[1.0498110175, 0.0000000000,-0.0000974845,],
	[-0.4959030231, 1.3733130458, 0.0982400361,],
	[0.0000000000, 0.0000000000, 0.9912520182]])

XYZ_2_AP1_MAT = np.array(
    [[1.6410233797, -0.3248032942, -0.2364246952,],
	[-0.6636628587,  1.6153315917,  0.0167563477,],
	[0.0117218943, -0.0082844420,  0.9883948585,]])

D65_2_D60_CAT = np.array([[ 1.01303   ,  0.00610531, -0.014971  ],
       [ 0.00769823,  0.998165  , -0.00503203],
       [-0.00284131,  0.00468516,  0.924507  ]])

sRGB_2_XYZ_MAT = np.array([[0.4124564, 0.3575761, 0.1804375],
       [0.2126729, 0.7151522, 0.072175 ],
       [0.0193339, 0.119192 , 0.9503041]])

XYZ_2_sRGB_MAT = np.array([[ 3.24096994, -1.53738318, -0.49861076],
       [-0.96924364,  1.8759675 ,  0.04155506],
       [ 0.05563008, -0.20397696,  1.05697151]])

D60_2_D65_CAT = np.array([[ 0.987224  , -0.00611327,  0.0159533 ],
       [-0.00759836,  1.00186   ,  0.00533002],
       [ 0.00307257, -0.00509595,  1.08168   ]])

AP0_2_XYZ_MAT = np.array([[ 9.52552396e-01,  0.00000000e+00,  9.36786000e-05],
       [ 3.43966450e-01,  7.28166097e-01, -7.21325464e-02],
       [ 0.00000000e+00,  0.00000000e+00,  1.00882518e+00]])

AP1_2_XYZ_MAT = np.array([[ 0.66245418,  0.13400421,  0.15618769],
       [ 0.27222872,  0.67408177,  0.05368952],
       [-0.00557465,  0.00406073,  1.0103391 ]])

AP0_2_AP1_MAT = np.array([[ 1.45143932, -0.23651075, -0.21492857],
       [-0.07655377,  1.1762297 , -0.09967593],
       [ 0.00831615, -0.00603245,  0.9977163 ]])


AP1_RGB2Y =np.array([
	0.2722287168,
	0.6740817658,
	0.0536895174]
)

def rgb_2_saturation(rgb):
	minrgb = rgb.min()
	maxrgb = rgb.max()
	return ( max( maxrgb, 1e-10 ) - max( minrgb, 1e-10 ) ) / max( maxrgb, 1e-2 )

def rgb_2_yc(rgb, ycRadiusWeight=1.75):
    r, g, b = rgb[...,0],rgb[...,1],rgb[...,2]
    chroma = np.sqrt(b*(b-g) + g*(g-r) + r*(r-b))
    return (b + g + r + ycRadiusWeight * chroma) / 3.


def sigmoid_shaper(x):
	t = max( 1 - abs( 0.5 * x ), 0 )
	y = 1 + np.sign(x) * (1 - t*t)
	return 0.5 * y

def glow_fwd(ycIn, glowGainIn, glowMid):
   glowGainOut = glowGainIn * (glowMid / ycIn - 0.5)
   first = np.where(ycIn <= 2./3. * glowMid)
   second = np.where((ycIn > 2./3. * glowMid)&(ycIn >= 2 * glowMid))
   glowGainOut[first] = glowGainIn
   glowGainOut[second] = 0
   return glowGainOut

def rgb_2_hue(rgb):
    hue = (180. / np.pi) * np.arctan2(np.sqrt(3.0)*(rgb[...,1] - rgb[...,2]), 2 * rgb[...,0] - rgb[...,1] - rgb[...,2])
    hue[np.where((rgb[...,0]==rgb[...,1])&(rgb[...,1]==rgb[...,2]))] =0
    hue[np.where(hue<0)] += 360
    return np.clip( hue, 0, 360)

def center_hue(hue,centerH):
	hueCentered = hue - centerH
	hueCentered[np.where(hueCentered < -180.)] +=360 
	hueCentered[np.where(hueCentered > 180.)] -=360
	return hueCentered

def smoothstep(A, B, X):
    A_pos = np.where(X < A)
    B_pos = np.where((X >= B))
    InterpFraction = (X - A) / (B - A)
    InterpFraction = InterpFraction * InterpFraction * (3. - 2.* InterpFraction)
    InterpFraction[A_pos] = 0
    InterpFraction[B_pos] = 1
    return InterpFraction

def Square(A):
    return A*A

sRGB_2_AP0 = XYZ_2_AP0_MAT*( D65_2_D60_CAT*sRGB_2_XYZ_MAT )
sRGB_2_AP1 = XYZ_2_AP1_MAT * ( D65_2_D60_CAT*sRGB_2_XYZ_MAT ) 

AP0_2_sRGB =  XYZ_2_sRGB_MAT * ( D60_2_D65_CAT*AP0_2_XYZ_MAT ) 
AP1_2_sRGB = XYZ_2_sRGB_MAT * ( D60_2_D65_CAT*AP1_2_XYZ_MAT ) 
	
AP0_2_AP1 = XYZ_2_AP1_MAT *AP0_2_XYZ_MAT 
AP1_2_AP0 = XYZ_2_AP0_MAT * AP1_2_XYZ_MAT 
LinearColor =  np.random.rand(256, 256, 3)
ColorAP1 = np.random.rand(256, 256, 3)
RRT_GLOW_GAIN = 0.05
RRT_GLOW_MID = 0.08
# np.dot
ColorAP1 = np.dot(LinearColor,sRGB_2_AP1)
ColorAP0 = np.dot(ColorAP1,AP1_2_AP0)
# glow
saturation = rgb_2_saturation(ColorAP0)
ycIn = rgb_2_yc( ColorAP0 )
s = sigmoid_shaper( (saturation - 0.4) / 0.2)
addedGlow = 1 + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID)
#Red modifier
RRT_RED_SCALE = 0.82
RRT_RED_PIVOT = 0.03
RRT_RED_HUE = 0
RRT_RED_WIDTH = 135
hue = rgb_2_hue( ColorAP0 )
centeredHue = center_hue( hue, RRT_RED_HUE )
hueWeight = Square( smoothstep( 0, 1, 1 - abs( 2 * centeredHue / RRT_RED_WIDTH ) ) )
ColorAP0[...,0] += hueWeight * saturation * (RRT_RED_PIVOT - ColorAP0[...,0]) * (1. - RRT_RED_SCALE)


# // Use ACEScg primaries as working space
WorkingColor = np.dot(ColorAP0,AP0_2_AP1_MAT)
WorkingColor[np.where(WorkingColor<0)] = 0
tmp = np.dot(WorkingColor,AP1_RGB2Y)[...,None].repeat(3,2)
WorkingColor = colour.utilities.lerp(tmp, WorkingColor, 0.96 )
	
ToeScale		= 1 + BlackClip - Toe
ShoulderScale	= 1 + WhiteClip - Shoulder
InMatch = 0.18
OutMatch = 0.18
bt = ( OutMatch + BlackClip ) / ToeScale - 1
ToeMatch = np.log10( InMatch ) - 0.5 * np.log( (1+bt)/(1-bt) ) * (ToeScale / Slope)
StraightMatch = ( 1 - Toe ) / Slope - ToeMatch
ShoulderMatch = Shoulder / Slope - StraightMatch
LogColor = np.log10( WorkingColor )
StraightColor = Slope * ( LogColor + StraightMatch )
ToeColor= (-BlackClip ) + (2 *ToeScale) / ( 1 + np.exp( (-2 * Slope /ToeScale) * ( LogColor - ToeMatch ) ) )
ShoulderColor	= ( 1 + WhiteClip ) - (2 * ShoulderScale) / ( 1 + np.exp( ( 2 * Slope / ShoulderScale) * ( LogColor - ShoulderMatch ) ) )
ToeColor[np.where(LogColor>ToeMatch)] = StraightColor[np.where(LogColor>ToeMatch)]
ShoulderColor[np.where(LogColor > ShoulderMatch)] = StraightColor[np.where(LogColor > ShoulderMatch)]
t = np.clip( ( LogColor - ToeMatch ) / ( ShoulderMatch - ToeMatch ) ,0,1)
t = 1 - t if ShoulderMatch < ToeMatch else t
t = (3-2*t)*t*t
ToneColor = colour.utilities.lerp( ToeColor, ShoulderColor, t )
ToneColor = colour.utilities.lerp( np.dot(ToneColor, AP1_RGB2Y )[...,None].repeat(3,2), ToneColor, 0.93 )
res = np.clip(ToneColor,0,None)