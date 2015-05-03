#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
using namespace std;
//#define DEBUG
typedef unsigned char uint8;
// small value, used to avoid division by zero
#define eps 0.0001
int round(float a)
{
    float tmp=a-(int)a;
    if(tmp>=0.5)
        return (int)a+1;
    else 
        return (int)a;
}

// unit vectors used to compute gradient orientation
double uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
double vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// main function:
// takes a double color image and a bin size 
// returns HOG features
void process1(double *im,int width,int height,int sbin) {

	// memory for caching orientation histograms & their norms
	int blocks[2];
	blocks[0] = (int)round((double)height/(double)sbin);//
	blocks[1] = (int)round((double)width/(double)sbin);//
	double *hist = (double *)malloc(blocks[0]*blocks[1]*18*sizeof(double));
	double *norm = (double *)malloc(blocks[0]*blocks[1]*sizeof(double));

	// memory for HOG features
	int out[3];
	out[0] = max(blocks[0]-2, 0);
	out[1] = max(blocks[1]-2, 0);
	out[2] = 27+4;
	double *feat = (double *)malloc(out[0]*out[1]*out[2]*sizeof(double));
  
	int visible[2];
	visible[0] = blocks[0]*sbin;
	visible[1] = blocks[1]*sbin;
	FILE *fp;
	for(int i=0;i<blocks[0]*blocks[1]*18;i++)
	{
		hist[i]=0;
	}
	for(int i=0;i<blocks[0]*blocks[1];i++)
	{
		norm[i]=0;
	}

	for (int x = 1; x < visible[0]-1; x++) 
	{
		for (int y = 1; y < visible[1]-1; y++)
		{
            // first color channel
			//cout<<*im<<endl;
			//cout<<*(im+1)<<endl;
			//cout<<*(im+2)<<endl;
            double *s = im + min(x, height-2)*width + min(y, width-2);
			//cout<<*s<<endl;
            double dx = *(s+1) - *(s-1);
			//cout<<dx<<endl;
            double dy = *(s+width) - *(s-width);
			//cout<<dy<<endl;
            double v = dx*dx + dy*dy;

            // second color channel
            s += width*height;
            double dx2 = *(s+1) - *(s-1);
            double dy2 = *(s+width) - *(s-width);
            double v2 = dx2*dx2 + dy2*dy2;

            // third color channel
            s += width*height;
            double dx3 = *(s+1) - *(s-1);
            double dy3 = *(s+width) - *(s-width);
            double v3 = dx3*dx3 + dy3*dy3;

            // pick channel with strongest gradient 
            if (v2 > v) 
            {
                v = v2;
                dx = dx2;
                dy = dy2;
            } 
            if (v3 > v)
            {
                v = v3;
                dx = dx3;
                dy = dy3;
            }

            // snap to one of 18 orientations

            double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < 9; o++)
            {
                //sin^2+cos^2=1
                //max cos*dx+sin*dy
				//uu->cos
				//vv->sin
                double dot = uu[o]*dx + vv[o]*dy;
                if (dot > best_dot) 
                {
                    best_dot = dot;
                    best_o = o;
                } else if (-dot > best_dot) 
                {
                    best_dot = -dot;
                    best_o = o+9;
                }
            }

            // add to 4 histograms around pixel using linear interpolation
            double xp = ((double)x+0.5)/(double)sbin - 0.5;
            double yp = ((double)y+0.5)/(double)sbin - 0.5;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
            double vx0 = xp-ixp;
            double vy0 = yp-iyp;
            double vx1 = 1.0-vx0;
            double vy1 = 1.0-vy0;
			
            v = sqrt(v);
            //lefr up
            if (ixp >= 0 && iyp >= 0)
            {
                *(hist + ixp*blocks[1] + iyp + best_o*blocks[0]*blocks[1]) += vx1*vy1*v;
				//printf("%d ",ixp*blocks[1] + iyp + best_o*blocks[0]*blocks[1]);
				//printf("%lf ",vx0*vy0*v);
            }
            //right up
            if (ixp+1 < blocks[0] && iyp >= 0) 
            {
                *(hist + (ixp+1)*blocks[1] + iyp + best_o*blocks[0]*blocks[1]) += vx0*vy1*v;
				//printf("%d ",(ixp+1)*blocks[1] + iyp + best_o*blocks[0]*blocks[1]);
				//printf("%lf ",vx0*vy0*v);
            }
            //left down
            if (ixp >= 0 && iyp+1 < blocks[1])
            {
                *(hist + ixp*blocks[1] + (iyp+1) + best_o*blocks[0]*blocks[1]) += vx1*vy0*v;
				//printf("%d ",ixp*blocks[1] + (iyp+1) + best_o*blocks[0]*blocks[1]);
				//printf("%lf ",vx0*vy0*v);
            }
            //right down
            if (ixp+1 < blocks[0] && iyp+1 < blocks[1]) 
            {
                *(hist + (ixp+1)*blocks[1] + (iyp+1) + best_o*blocks[0]*blocks[1]) += vx0*vy0*v;
				//printf("%d ",(ixp+1)*blocks[1] + (iyp+1) + best_o*blocks[0]*blocks[1]);
				//printf("%lf ",vx0*vy0*v);
            }
			//printf("o=%lf ",best_o);
			//printf("\n");
		}
	}
#ifdef DEBUG
	fp = fopen("hist.txt","w");
	for(int i=0;i<blocks[0]*blocks[1];i++)
	{
		for(int j=0;j<18;j++)
			fprintf(fp,"%lf ",hist[i*18+j]);
		fprintf(fp,"\n");
	}
	fclose(fp);
#endif
	// compute energy in each block by summing over orientations
	
	for (int o = 0; o < 9; o++) 
	{
		double *src1 = hist + o*blocks[0]*blocks[1];
		double *src2 = hist + (o+9)*blocks[0]*blocks[1];
		double *dst = norm;
		double *end = norm + blocks[1]*blocks[0];
		while (dst < end)
		{
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}
	#ifdef DEBUG
	fp = fopen("norm.txt","w");
	for(int i=0;i<blocks[0]*blocks[1];i++)
	{
			fprintf(fp,"%lf\n",norm[i]);
	}
	fclose(fp);
#endif
	// compute features
	for (int x = 0; x < out[0]; x++) 
	{
		for (int y = 0; y < out[1]; y++)
		{
			double *dst = feat + x*out[1] + y;      
			double *src, *p, n1, n2, n3, n4;
	
			p = norm + (x+1)*blocks[1] + y+1;
			n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
			p = norm + (x+1)*blocks[1] + y;
			n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
			p = norm + x*blocks[1] + y+1;
			n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
			p = norm + x*blocks[1] + y;
			n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);

			double t1 = 0;
			double t2 = 0;
			double t3 = 0;
			double t4 = 0;

			// contrast-sensitive features
			src = hist + (x+1)*blocks[1] + (y+1);
			for (int o = 0; o < 18; o++)
			{
				double h1 = min(*src * n1, 0.2);
				double h2 = min(*src * n2, 0.2);
				double h3 = min(*src * n3, 0.2);
				double h4 = min(*src * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);//sum
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				dst += out[0]*out[1];
				src += blocks[0]*blocks[1];
			}

			// contrast-insensitive features
			src = hist + (x+1)*blocks[1] + (y+1);
			for (int o = 0; o < 9; o++)
			{
                double sum = *src + *(src + 9*blocks[0]*blocks[1]);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                dst += out[0]*out[1];
                src += blocks[0]*blocks[1];
            }

            // texture features
            *dst = 0.2357 * t1;
            dst += out[0]*out[1];
            *dst = 0.2357 * t2;
            dst += out[0]*out[1];
            *dst = 0.2357 * t3;
            dst += out[0]*out[1];
            *dst = 0.2357 * t4;
		}	
	}
	#ifdef DEBUG
	fp = fopen("fea.txt","w");
	for(int i=0;i<out[0]*out[1]*out[2];i++)
	{
		fprintf(fp,"%lf\n",feat[i]);
	}
	fclose(fp);
#endif

	free(hist);
	free(norm);

}

void process2(double *im,int width,int height,int sbin,double *GM,double *GO/*,double *GDOT*/) {

	// memory for caching orientation histograms & their norms
	int blocks[2];
	blocks[0] = (int)round((double)height/(double)sbin);//ÐÐ
	blocks[1] = (int)round((double)width/(double)sbin);//ÁÐ
	double *hist = (double *)malloc(blocks[0]*blocks[1]*18*sizeof(double));
	double *norm = (double *)malloc(blocks[0]*blocks[1]*sizeof(double));

	// memory for HOG features
	int out[3];
	out[0] = max(blocks[0]-2, 0);
	out[1] = max(blocks[1]-2, 0);
	out[2] = 27+4;
	double *feat = (double *)malloc(out[0]*out[1]*out[2]*sizeof(double));
  
	int visible[2];
	visible[0] = blocks[0]*sbin;
	visible[1] = blocks[1]*sbin;
	FILE *fp;
	for(int i=0;i<blocks[0]*blocks[1]*18;i++)
	{
		hist[i]=0;
	}
	for(int i=0;i<blocks[0]*blocks[1];i++)
	{
		norm[i]=0;
	}
	//fp=fopen("wieght.txt","w");
	for (int x = 1; x < visible[0]-1; x++) 
	{
		for (int y = 1; y < visible[1]-1; y++)
		{
            // first color channel
			//cout<<*im<<endl;
			//cout<<*(im+1)<<endl;
			//cout<<*(im+2)<<endl;
            double *s = im + min(x, height-2)*width + min(y, width-2);//location in im

            //double dx = *(s+1) - *(s-1);
            //double dy = *(s+width) - *(s-width);
            //double v = dx*dx + dy*dy;

			int dx =*(s+1) - *(s-1);
			int dy = *(s+width) -*(s-width);
			double v = GM[(dx+255)*511+(dy+255)];

            // second color channel
            s += width*height;
            //double dx2 = *(s+1) - *(s-1);
            //double dy2 = *(s+width) - *(s-width);
            //double v2 = dx2*dx2 + dy2*dy2;

			int dx2 =*(s+1) - *(s-1);
			int dy2 =*(s+width) - *(s-width);
			double v2 = GM[(dx2+255)*511+(dy2+255)];

            // third color channel
            s += width*height;
            //double dx3 = *(s+1) - *(s-1);
            //double dy3 = *(s+width) - *(s-width);
            //double v3 = dx3*dx3 + dy3*dy3;

			int dx3 = *(s+1) - *(s-1);
			int dy3 = *(s+width) - *(s-width);
			double v3 =GM[(dx3+255)*511+(dy3+255)];

            // pick channel with strongest gradient
            if (v2 > v) 
            {
                v = v2;
                dx = dx2;
                dy = dy2;
            } 
            if (v3 > v)
            {
                v = v3;
                dx = dx3;
                dy = dy3;
            }

            // snap to one of 18 orientations
			#if 0
			double best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 9; o++)
			{
				//sin^2+cos^2=1
				//max cos*dx+sin*dy
				//uu->cos
				//vv->sin
				double dot = uu[o]*dx + vv[o]*dy;
				if (dot > best_dot) 
				{
					best_dot = dot;
					best_o = o;
				} else if (-dot > best_dot) 
				{
					best_dot = -dot;
					best_o = o+9;
				}
			}
			#endif
			//double best_dot = GDOT[(dx+255)*511+(dy+255)];
			int best_o =GO[((int)dx+255)*511+((int)dy+255)];
            // add to 4 histograms around pixel using linear interpolation
            //double xp = ((double)x+0.5)/(double)sbin - 0.5;
            //double yp = ((double)y+0.5)/(double)sbin - 0.5;
			double xp = (double)x/(double)sbin - 0.4375;
            double yp = (double)y/(double)sbin - 0.4375;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
			//fprintf(fp,"x=%d,xp=%lf,ixp=%d,y=%d,yp=%lf,iyp=%d\n",x,xp,ixp,y,yp,iyp);
            double vx0 = xp-ixp;
            double vy0 = yp-iyp;
            double vx1 = 1.0-vx0;
            double vy1 = 1.0-vy0;
			
           // v = sqrt(v);
            //left up
            if (ixp >= 0 && iyp >= 0)
            {
                *(hist + ixp*blocks[1] + iyp + best_o*blocks[0]*blocks[1]) += vx1*vy1*v;
				//printf("%d ",ixp*blocks[1] + iyp + best_o*blocks[0]*blocks[1]);
				//printf("%lf ",vx0*vy0*v);
            }
            //right up
            if (ixp+1 < blocks[0] && iyp >= 0) 
            {
                *(hist + (ixp+1)*blocks[1] + iyp + best_o*blocks[0]*blocks[1]) += vx0*vy1*v;
				//printf("%d ",(ixp+1)*blocks[1] + iyp + best_o*blocks[0]*blocks[1]);
				//printf("%lf ",vx0*vy0*v);
            }
            //left down
            if (ixp >= 0 && iyp+1 < blocks[1])
            {
                *(hist + ixp*blocks[1] + (iyp+1) + best_o*blocks[0]*blocks[1]) += vx1*vy0*v;
				//printf("%d ",ixp*blocks[1] + (iyp+1) + best_o*blocks[0]*blocks[1]);
				//printf("%lf ",vx0*vy0*v);
            }
            //right down
            if (ixp+1 < blocks[0] && iyp+1 < blocks[1]) 
            {
                *(hist + (ixp+1)*blocks[1] + (iyp+1) + best_o*blocks[0]*blocks[1]) += vx0*vy0*v;
				//printf("%d ",(ixp+1)*blocks[1] + (iyp+1) + best_o*blocks[0]*blocks[1]);
				//printf("%lf ",vx0*vy0*v);
            }
			//printf("o=%lf ",best_o);
			//printf("\n");
		}
	}
	//fclose(fp);
#ifdef DEBUG
	fp = fopen("hist1.txt","w");
	for(int i=0;i<blocks[0]*blocks[1];i++)
	{
		for(int j=0;j<18;j++)
			fprintf(fp,"%lf ",hist[i*18+j]);
		fprintf(fp,"\n");
	}
	fclose(fp);
#endif
	// compute energy in each block by summing over orientations
	//sum((v(oi)+v(oi+9))^2),oi=0...8
	for (int o = 0; o < 9; o++) 
	{
		double *src1 = hist + o*blocks[0]*blocks[1];
		double *src2 = hist + (o+9)*blocks[0]*blocks[1];
		double *dst = norm;
		double *end = norm + blocks[1]*blocks[0];
		while (dst < end)
		{
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}
	#ifdef DEBUG
	fp = fopen("norm1.txt","w");
	for(int i=0;i<blocks[0]*blocks[1];i++)
	{
			fprintf(fp,"%lf\n",norm[i]);
	}
	fclose(fp);
#endif
	// compute features
	for (int x = 0; x < out[0]; x++) 
	{
		for (int y = 0; y < out[1]; y++)
		{
			double *dst = feat + x*out[1] + y;      
			double *src, *p, n1, n2, n3, n4;
	
			p = norm + (x+1)*blocks[1] + y+1;//right down constrain insensitive sum
			n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
			p = norm + (x+1)*blocks[1] + y;//right
			n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
			p = norm + x*blocks[1] + y+1;//down
			n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);
			p = norm + x*blocks[1] + y;//self
			n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[1]) + *(p+blocks[1]+1) + eps);

			double t1 = 0;
			double t2 = 0;
			double t3 = 0;
			double t4 = 0;

			// contrast-sensitive features
			src = hist + (x+1)*blocks[1] + (y+1);
			for (int o = 0; o < 18; o++)
			{
				double h1 = min(*src * n1, 0.2);//
				double h2 = min(*src * n2, 0.2);
				double h3 = min(*src * n3, 0.2);
				double h4 = min(*src * n4, 0.2);
				*dst = 0.5 * (h1 + h2 + h3 + h4);//sum
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				dst += out[0]*out[1];
				src += blocks[0]*blocks[1];
			}

			// contrast-insensitive features
			src = hist + (x+1)*blocks[1] + (y+1);
			for (int o = 0; o < 9; o++)
			{
                double sum = *src + *(src + 9*blocks[0]*blocks[1]);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                dst += out[0]*out[1];
                src += blocks[0]*blocks[1];
            }

            // texture features
            *dst = 0.2357 * t1;
            dst += out[0]*out[1];
            *dst = 0.2357 * t2;
            dst += out[0]*out[1];
            *dst = 0.2357 * t3;
            dst += out[0]*out[1];
            *dst = 0.2357 * t4;
		}	
	}
	#ifdef DEBUG
	fp = fopen("fea1.txt","w");
	for(int i=0;i<out[0]*out[1]*out[2];i++)
	{
		fprintf(fp,"%lf\n",feat[i]);
	}
	fclose(fp);
#endif

	free(hist);
	free(norm);

}

int writeraw(char *fname,uint8 *pin, int width, int height)
{
	int datsize =width *height;
	FILE *fp = fopen(fname,"wb");
	if (fp==NULL)
	{
		printf("can't write file\n");
		return 1;
	}

	fwrite(pin,sizeof(uint8),datsize,fp);
	fclose(fp);

	return 0;
}

int readraw(char *fname,uint8 *img, int width, int height)
{
	int datsize = height *width;
	FILE *fp = fopen(fname,"rb");
	if (fp==NULL)
	{
		printf("can't open file\n");
		return 1;
	}

	fread(img,sizeof(uint8),datsize,fp);
	fclose(fp);
	return 0;
}

int main()
{
	//char* fin1=".\\img\\t18.raw";
	//int width=143;
	//int height=129;
	char* fin1=".\\img\\5.raw";
	int width=2592;
	int height=1936;
	uint8 * in1=(uint8 *)malloc(sizeof(uint8)*width*height);
	double * in2=(double *)malloc(sizeof(double)*width*height*3);
	readraw(fin1,in1,width,height);
	for(int i=0;i<width*height;i++)
	{
		in2[i]=in1[i];
		in2[i+height*width]=in1[i];
		in2[i+2*height*width]=in1[i];
	}
	double start,end,cost;
	start=clock();
	process1(in2,width,height,8);
	end=clock();
	cost=end-start;
	printf("cost1=%lf\n",cost);
	double * GM=(double *)malloc(sizeof(double)*511*511);
	for(int i=-255;i<=255;i++)
		for(int j=-255;j<=255;j++)
			GM[(i+255)*511+(j+255)]=sqrt(i*i+j*j);
	double * GO=(double *)malloc(sizeof(double)*511*511);
	double * GDOT=(double *)malloc(sizeof(double)*511*511);
	for(int i=-255;i<=255;i++)
		for(int j=-255;j<=255;j++)
		{
			double best_dot = 0;
            int best_o = 0;
            for (int o = 0; o < 9; o++)
            {
                //sin^2+cos^2=1
                //max cos*dx+sin*dy
				//uu->cos
				//vv->sin
                double dot = uu[o]*i + vv[o]*j;
                if (dot > best_dot) 
                {
                    best_dot = dot;
                    best_o = o;
                } else if (-dot > best_dot) 
                {
                    best_dot = -dot;
                    best_o = o+9;
                }
            }
			GO[(i+255)*511+(j+255)]=best_o;
			GDOT[(i+255)*511+(j+255)]=best_dot;
		}
	start=clock();
	process2(in2,width,height,8,GM,GO/*,GDOT*/);
	end=clock();
	cost=end-start;
	printf("cost2=%lf\n",cost);
	free(in1);
	free(in2);
	free(GM);
	free(GO);
	free(GDOT);
	getchar();
	return 0;
}