import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileWriter;
import java.io.File;
import java.util.*;

public class logHmm {
    
    int k1; //total dimension 
    int k2; //short dimension;
    int l; // transitions
    int[] t; //times steps
    int n; //number of data points
    
    double prior = 0;
    
    double logLik;
    double threshold = .00001;
    
    
    double[][] x;
    
    /**
     * E step Variable
     */
    double[][][] alpha; //recursive alpha
    double[][][] beta; //recursive beta
    double[][][] emit; //posterior evidence
    double[][] xiSum; //pairwise transition 
    double[][][] gamma; //individual propbabilitie
    
    /**
     * M Step Variables
     */
    double[][] trans; //transition matrix
    double[][] thetaX; //x parameters
    
    public static void main(String[] args) {
        
        logHmm obj = new logHmm();
        obj.run();
        
    }
    
    
    // inference for the EM algorithm
    public void infer(int nn, int kk1, int kk2, int tt[])
    {
        init(nn,kk1, kk2, tt);
        paramInit();
        int count = 0;
        do
        {
            eStep();
            count ++;
            mStep();
            
            if (count%100 == 0)
                System.out.println("Log Likelihood: " + logLik); 
            
        }
        while (!converged());
        System.out.println("Log Likelihood: " + logLik); 
        printParams();
        
    }
    
    public void printParams()
    {
        for (int i = 0; i < k1; i++)
        {
            System.out.println("State " + i + ": " + thetaX[i][2] + " " + thetaX[i][1] +
                               thetaX[i][2] ); 
            for (int j = 0; j < k1; j++)
            {
                System.out.println("State " + i + " to state " + j + " " + trans[i][j]); 
            }
        }
        String out = "";
        for (int i = 0; i < k1; i++)
            out += "," + i;
        out+="\n";
         for (int i = 0; i < k1; i++)
         {
             out+= "" + i;
             for (int j = 0; j < k1; j++)
                  out += "," + (100*Math.exp(trans[i][j]));
             out+= "\n";
         }
         
         System.out.println(out);
        
        
    }
    
    
    private int max(int[] arr)
    {
        int max = 0;
        for (int i = 0; i<arr.length; i++)
            if (arr[i]>max) max = arr[i];
       
        return max;
    }
    
    public void init(int nn, int kk1,  int kk2, int[] tt)
    {
        k1 = kk1*kk2;
        k2 = kk2;
        t = tt;
        n = nn;
  
        int tlen = max(t);
        
        alpha = new double[n][tlen][k1]; //recursive alpha
        beta = new double[n][tlen][k1]; ; //recursive beta
        emit = new double[n][tlen][k1]; //posterior evidence
        gamma = new double[n][tlen][k1];; //individual propbabilitie
        xiSum = new double[k1][k1]; //pairwise transition 
        
        trans = new double[k1][k1]; //transition matrix
        thetaX = new double[k1][3]; //x parameters
        
    }
    
    public boolean okay(int i, int j)
    {
       return true;
       /**
        if (i == j)
            return true;
        // move to next state in subcycle
        if (i%k2 < k2 - 1 &&  j == i+1)
            return true;
        // go to begining of new big state
        if (j % k2 == 0) 
            return true;
        
        return false;
        **/
    }
    
    public double setXiSum(int i, int j)
    {
     
     if (okay(i,j))
           return 4 + Math.random();
      return 0;
    }
    
    
    // initialize all parameters for the EM algorithm
    public void paramInit()
    {
        logLik = -99999;
        
        double[][] init = new double[n][k1];
        double sum = 0;
        
        for (int ni = 0; ni<n;ni++)
        {
            sum = 0;
            for (int i = 0; i < k1; i++)
            {
                double temp = 3 + Math.random()/3;
                sum+= temp;
                init[ni][i] = temp;
            }
            for (int i = 0; i < k1; i++)
            {   
                gamma[ni][0][i] = init[ni][i]/sum;

            }
            assert(isNormalized(ni,0,gamma));
        }
        
        for (int i = 0; i < k1; i++)
        {
            
            for (int j = 0; j<k1; j++)
            {
                xiSum[i][j] = setXiSum(i,j);
            }
            
            thetaX[i][0] = 1;
            thetaX[i][1] = Math.random() - .5;
            thetaX[i][2] = .3;

            
        }
        

        
        //normalizes xiSum;
        makeTrans();
        // posterior distribution
        getEmit();
    }
        
    
    
    // returns posterior probability of x~N(mu,sigmaSq) up to
    // constant of 1/\sqrt{2\pi}
    public double normal(double x, double mu, double sigmaSq){return Math.exp(logNormal(x,mu,sigmaSq));}
    public double logNormal(double x, double mu, double sigmaSq){return -1*(x-mu)*(x-mu)/(2*sigmaSq) - Math.log(sigmaSq)/2;}
    
    //update loglikelihood, return true if convergence achieved
    public boolean converged()
    {
        double lik = logLik;
        logLik = likelihood();
        return (Math.abs(lik-logLik)<threshold);
    }
    
    
    //compute log likelihood()
    public double likelihood()
    {
        double lik = 0; 
        for (int i = 0; i < k1; i++)
        {
            assert(!Double.isNaN(lik));
            for (int j = 0; j<k1;j++)
            {
                //if (okay(i,j))
                 lik+=xiSum[i][j]*trans[i][j];
            }
            assert(!Double.isNaN(lik));
            for (int ni = 0; ni<n;ni++)
            {
                lik+=gamma[ni][0][i]*Math.log(gamma[ni][0][i]);
                for (int ti =1; ti<t[ni]; ti++)
                {
                    lik+=gamma[ni][ti][i]*emit[ni][ti][i];
                    
                    assert(!Double.isNaN(lik));
                }
            }
        }
        return lik;
    }
    
    //M Step in EM algorithm
    public void mStep()
    {
        makeTrans();
        updateParameters(thetaX,x);
        getEmit();
    }
    
    public double hmmLog(double a)
    {
        if (a==0) return Double.NEGATIVE_INFINITY;
        return Math.log(a);
    }
    
    
    //Make the Transition Matrix
    public void makeTrans()
    {
        double[][] tempTrans = new double[k1][k1];

        for (int i=0;i<k1;i++)
        {
            double sum = 0;
            for (int j = 0; j<k1;j++)
                sum += xiSum[i][j];
            
            for (int j = 0; j < k1;j++)
            {
                
                 // if (okay(i,j))
                tempTrans[i][j] = xiSum[i][j]/sum;
                trans[i][j] = hmmLog(xiSum[i][j]/sum);
            }
            assert(isNormalized(i,tempTrans));
        }
        
        
    }
    
    //Estimates Parameters for an HMM. 
    public void updateParameters(double[][] output, double[][] data)
    {
        for (int i = 0; i<k1; i++)
        {
            double[][] system = new double[2][3]; 
            
            // set the sytem to 0
            for (int j1 = 0; j1 < 2; j1++)
                for (int j2 = 0; j2<3; j2++)
                system[j1][j2] = 0;
            for(int ni = 0; ni < n; ni++)
            for (int ti = 1; ti < t[ni]; ti++)
            {
                system[0][0] += gamma[ni][ti][i]*data[ni][ti]; 
                system[0][1] += gamma[ni][ti][i]*data[ni][ti-1]; 
                system[0][2] += gamma[ni][ti][i];
                
                system[1][0] += gamma[ni][ti][i]*data[ni][ti] * data[ni][ti-1];
                system[1][1] += gamma[ni][ti][i]*data[ni][ti-1] * data[ni][ti-1];
                system[0][2] += gamma[ni][ti][i]*data[ni][ti-1]; 
            }
            
            double[] solution = solveSystem(system);
            
            double mu = solution[0];
            double a = solution[1];
            double sigma = 0;
            
            for(int ni = 0; ni < n; ni++)
            for (int ti=1;ti<t[ni];ti++)
                sigma += gamma[ni][ti][i]* Math.pow(data[ni][ti]-a*data[ni][ti-1]-mu,2);
            sigma /= system[0][2];
            
            output[i][0] = a;
            output[i][1] = mu;
            output[i][2] = sigma;
            
        }
    }
    
    private double[] solveSystem(double[][] system)
    {
        
        double[] output = new double[2];
        
        double a = system[0][1];
        double b = system[0][2];
        double c = system[1][1];
        double d = system[1][2];
        
        double det = a*d - (b*c);
        output[0] = (d*system[0][0]-b*system[1][0])/det;
        output[1] = (a*system[1][0] - c*system[0][0] )/det;
        
        return output;
        
    }
    
    // compute posterior probabilities
    public void getEmit()
    {
        for (int ni = 0; ni < n; ni++)
        {
            double mu; //temporary mean
            for(int i1=0; i1<k1; i1++)
                emit[ni][0][i1] = 0;
            
            for (int ti=1; ti<t[ni]; ti++)
                for(int i=0; i<k1; i++)
            {
                mu = thetaX[i][0]*x[ni][ti-1]
                    -thetaX[i][1];
                emit[ni][ti][i]=logNormal(x[ni][ti],mu,thetaX[i][2]);
                
            }
        }
    }
    
    
    public void eStep()
    {
        getAlpha();
        getBeta();
        getXiSum();
        getGamma();
        
        //sanityCheck();
        
    }
    
    public void getAlpha()
    {
        
        //set up alpha_1
        for (int ni = 0; ni < n; ni ++)
            for (int i1=0; i1<k1;i1++)
            alpha[ni][0][i1] = hmmLog(gamma[ni][0][i1]);
        
        for (int ni = 0; ni < n; ni++)
            for (int ti =1; ti<t[ni]; ti++)
        {
            for  (int j= 0; j<k1 ;j++)
            {
                double alpha_t = Double.NEGATIVE_INFINITY;
                for(int i=0; i<k1; i++)
                {
                    
                    double log_sum = alpha[ni][ti-1][i] + trans[i][j] + emit[ni][ti][i];
                    alpha_t = logSum(alpha_t,log_sum);
                }
                alpha[ni][ti][j] = alpha_t;
            }
            logNormalize(ni,ti,alpha);   
        }
        
    }
    
    
    public void getBeta()
    {
        for (int ni = 0; ni< n; ni++)
        {
            for (int i = 0; i < k1; i++)
                beta[ni][t[ni]-1][i] = 0;
            
            for (int ti = t[ni]-2; ti>=0;ti--)
                for (int i = 0; i < k1; i++)
                beta[ni][ti][i] = Double.NEGATIVE_INFINITY;
            
            for (int ti = t[ni]-2; ti>=0;ti--)
            {
                for (int i = 0; i < k1; i++)
                {
                    double beta_t = Double.NEGATIVE_INFINITY;
                    for (int j = 0; j < k1; j++)
                    {
                        //if (!okay(i,j)) continue;
                        
                        double log_sum = trans[i][j] + emit[ni][ti+1][j] + beta[ni][ti+1][j];
                        beta_t = logSum(beta_t, log_sum);
                    }
                    beta[ni][ti][i] = beta_t;
                }
                logNormalize(ni,ti,beta);
            }
        }
    }
    
    public void getGamma()
    {
        for (int ni = 0; ni < n; ni++)
        {
            
            double value; 
            
            for (int ti = 0; ti < t[ni]; ti++)
            {
                double sum = Double.NEGATIVE_INFINITY; 
                double[] gt = new double[k1];
                for (int i= 0; i < k1; i++)
                {
                    value = alpha[ni][ti][i]+beta[ni][ti][i];
                    sum = logSum(sum,value);
                    gt[i] = value;
                }
                for (int i=0;i<k1;i++)
                {
                    gamma[ni][ti][i] = Math.exp(gt[i]-sum);
                }
                //assert(isNormalized(ni,ti,gamma));
            }
        }
    }
    
    public void getXiSum()
    {
        // clear XiSum
        for (int i = 0; i < k1; i++)
            for (int j = 0; j < k1; j++)
            xiSum[i][j] = prior;
        
        double value;
        
        for (int ni = 0; ni < n; ni++)
        for (int ti = 0; ti < t[ni]-1; ti++)
        {
            double[][] log_xi = new double[k1][k1];
            
            for (int i = 0; i < k1; i++)
            {
                double sum = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < k1; j++)
                {
                    //if (!okay(i,j)) continue;
                    
                    value = alpha[ni][ti][i]+beta[ni][ti+1][j]+trans[i][j]+emit[ni][ti+1][j];
                    
                    /**
                    assert(!Double.isNaN(trans[i][j]));
                    assert(!Double.isNaN(alpha[i][j]));
                    assert(!Double.isNaN(beta[i][j]));
                    assert(!Double.isNaN(emit[i][j]));
                    
                    assert(!Double.isInfinite(trans[i][j]));
                    assert(!Double.isInfinite(alpha[i][j]));
                    assert(!Double.isInfinite(beta[i][j]));
                    assert(!Double.isInfinite(emit[i][j]));
                    **/
                    
                    assert(value!=Double.NEGATIVE_INFINITY);
                    
                    log_xi[i][j] = value;
                    sum = logSum(sum,value);
                }
                
                
                double sum2 = 0;
                // exponentiate and normalize
                for (int j = 0; j < k1; j++)
                {
                    double val1 = Math.exp(log_xi[i][j]);
                    log_xi[i][j]= val1;
                    sum2+=val1;
                }
                
                //assert(isNormalized(i, log_xi));
                
                for (int j = 0; j < k1; j++)
                {
                    //if (!okay(i,j)) 
                      //  xiSum[i][j] = 0;
                    // else
                   xiSum[i][j] += log_xi[i][j]/sum2;
                }
                }
        }
    }

    public void run() 
    {  
        // input and output files
        String inputFile = "/Users/msimchowitz/Documents/COS424Data/oil.tsv";
        String transFile = "/Users/msimchowitz/Documents/COS424Data/trans.csv";
        String paramFile = "/Users/msimchowitz/Documents/COS424Data/params.csv";
        // settings for csv reader and writer
        BufferedReader br = null;
        String line = "";
        String csvSplitBy = "\t";
        boolean locHoldout = false;
        ArrayList<Double> oil = new ArrayList<Double>();
        ArrayList<ArrayList<Double>> dat = new ArrayList<ArrayList<Double>>();
        for (int i=0; i < 2; i++)
        {
            dat.add(new ArrayList<Double>() ) ;
        }
        
        
        try {
            br = new BufferedReader(new FileReader(inputFile));
            
            String us_num = "";
            
            while ((line = br.readLine()) != null) {
                
                String[] info = line.split(csvSplitBy);
                assert(info[0]!=null);
                for (int i=0; i < 2; i++)
                {
                    dat.get(i).add(Double.parseDouble(info[i]));
                    //System.out.println(""+ Double.parseDouble(info[i]));
                }
                
            }
            
            
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        
        int ttt = dat.get(0).size();
        int size = dat.size();
        
        x = new double[size][ttt];
        int[] tt = new int[size];
        
        for (int ni = 0; ni < size; ni++)
        {
            tt[ni] = dat.get(ni).size();
            for (int ti = 0; ti<ttt;ti++)
            {
                x[ni][ti]=dat.get(ni).get(ti);
            }
        }
      
        
        infer(size,3,1,tt);
        
        
         try {
            FileWriter tw = new FileWriter(transFile);
            FileWriter pw = new FileWriter(paramFile);
            
            for (int i = 0; i<k1;i++)
            {
                tw.append(",");
                tw.append("" + i);
            }
            tw.append("\n");
                
            for (int i = 0; i<k1;i++)
            {
                tw.append("" + i);
                for (int j = 0; j<k1;j++)
                {
                    tw.append(',');
                    tw.append("" + Math.exp(trans[i][j]));
                }
                tw.append('\n');
                
                // write parameter values
                pw.append("" + thetaX[i][0]);
                pw.append('\t');
                pw.append("" + thetaX[i][1]);
                pw.append('\t');
                pw.append("" + thetaX[i][2]);
                pw.append('\n');
                
            }
            
            
            
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    
    
    public double logSum(double a, double b)
    {
        return Math.log(Math.exp(b)+Math.exp(a));
    }
    
    public void logNormalize(int ni, int i, double[][][] arr)
    {
        double sum = Double.NEGATIVE_INFINITY;
        for(int j=0; j < arr[ni][i].length; j++)
            sum = logSum(sum,arr[ni][i][j]);
        
        for(int j=0; j < arr[ni][i].length; j++)
            arr[ni][i][j]-=sum;
    }
    
    public boolean isNormalized(int ni, int i, double[][][] arr)
    {
        double sum = 0;
        for (int j = 0; j < arr[ni][i].length;j++)
        {
            sum+=arr[ni][i][j];
        }
        if (Math.abs(sum-1)<.001)
            return true;
        return false;
    }
    
     public boolean isNormalized(int i, double[][] arr)
    {
        double sum = 0;
        for (int j = 0; j < arr[i].length;j++)
        {
            sum+=arr[i][j];
        }
        if (Math.abs(sum-1)<1)
            return true;
        return false;
    }
    
    
    
    }
