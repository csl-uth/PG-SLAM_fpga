#ifndef RGB_2_LAB_H
#define RGB_2_LAB_H

#define MIN_L 0.f
#define MAX_L 100.f

#define MIN_A -86.185
#define MAX_A 98.254

#define MIN_B -107.863
#define MAX_B 94.482

__host__ __device__  uchar3 lab2rgb(float3 lab)
{
    uchar3 ret;

    float y = (lab.x + 16) / 116;
    float x = lab.y / 500 + y;
    float z = y - lab.z / 200;
    float r, g, b;

    x = 0.95047 * ((x * x * x > 0.008856) ? x * x * x : (x - 16/116) / 7.787);
    y = 1.00000 * ((y * y * y > 0.008856) ? y * y * y : (y - 16/116) / 7.787);
    z = 1.08883 * ((z * z * z > 0.008856) ? z * z * z : (z - 16/116) / 7.787);

    r = x *  3.2406 + y * -1.5372 + z * -0.4986;
    g = x * -0.9689 + y *  1.8758 + z *  0.0415;
    b = x *  0.0557 + y * -0.2040 + z *  1.0570;

    r = (r > 0.0031308) ? (1.055 * pow(r, 1/2.4) - 0.055) : 12.92 * r;
    g = (g > 0.0031308) ? (1.055 * pow(g, 1/2.4) - 0.055) : 12.92 * g;
    b = (b > 0.0031308) ? (1.055 * pow(b, 1/2.4) - 0.055) : 12.92 * b;

    ret.x=(unsigned char)(max(0.f, min(1.f, r)) * 255);
    ret.y=(unsigned char)(max(0.f, min(1.f, g)) * 255);
    ret.z=(unsigned char)(max(0.f, min(1.f, b)) * 255);
    return ret;
}


__host__ __device__ float3 rgb2lab(uchar3 rgb)
{
    float3 ret;
    float r = ( (float)rgb.x) / 255.f;
    float g = ( (float)rgb.y) / 255.f;
    float b = ( (float)rgb.z) / 255.f;
    float x, y, z;

    r = (r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

    x = (x > 0.008856) ? pow(x, 1.f/3.f) : (7.787 * x) + 16.f/116.f;
    y = (y > 0.008856) ? pow(y, 1.f/3.f) : (7.787 * y) + 16.f/116.f;
    z = (z > 0.008856) ? pow(z, 1.f/3.f) : (7.787 * z) + 16.f/116.f;

    ret.x=(116 * y) - 16;
    ret.y=500 * (x - y);
    ret.z= 200 * (y - z);
    return ret;
}




#endif
