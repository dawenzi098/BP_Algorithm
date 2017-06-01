using System;
using System.IO;
using System.Text;
using System.Threading;

namespace Demo_03
{
    public class NNet
    {
        // number of input, hiden node, output
        public int input_Num;
        public int hide_Num;
        public int output_Num;
        public int sampleNum;
        public double standOut;
        public int rightNum;
        public int wrongNum;
        public double finalError;

        public double[] outputBates;
        public double[] hiddenBates;
        //public double[] error;
        public double[] inputError;

        // input data for each layer and output 
        Random R;
        double[] input_hide;
        double[] output_hide;
        double[] input_output;
        double[] output_output;


        //weight for each layer's input
        public double[,] weight_hidden;
        public double[,] weight_output;
        //public double[,] dweight_hidden;
        //public double[,] dweight_output;
        public double[,] rate;
        public double[,] inputrate;


        public NNet()
        {
            R = new Random(16);

            this.input_Num = 4;
            this.hide_Num = 4;
            this.output_Num = 1;
            this.sampleNum = 150;
            this.finalError = 0.0;
            this.rightNum = 0;
            this.wrongNum = 0;




            input_hide = new double[input_Num];
            output_hide = new double[hide_Num];
            input_output = new double[hide_Num];
            output_output = new double[output_Num];
            inputError = new double[hide_Num];
            rate = new double[hide_Num, output_Num];
            inputrate = new double[input_Num, hide_Num];

            weight_hidden = new double[input_Num, hide_Num];
            weight_output = new double[hide_Num, output_Num];

            outputBates = new double[output_Num];
            hiddenBates = new double[hide_Num];



            //initial all weight
            for (int i = 0; i < input_Num; i++)
            {
                for (int j = 0; j < hide_Num; j++)
                {
                    weight_hidden[i, j] = (R.NextDouble() * 2.0 - 1.0) / 2;
                }

            }

            for (int i = 0; i < hide_Num; i++)
            {
                for (int j = 0; j < output_Num; j++)
                {
                    weight_output[i, j] = (R.NextDouble() * 2.0 - 1.0) / 2;
                }
            }
        }


/**
 *  At this point, NN already initial, weight between input--hidden and hidden--output are all random initial
 * 
 * 
 * 
 * */
        public void train(double[,] fInput)
        {
             
            //find the max and min for each input
            double fMax = 0.0;
            double fMin = 999;

            double sMax = 0.0;
            double sMin = 999;

            double tMax = 0.0;
            double tMin = 999;

            double dMax = 0.0;
            double dMin = 999;

            System.Console.WriteLine("Start training---find max and min");
            //find min and max for each input
            for (int i = 0; i < sampleNum; i++)
            {
                //f
                if (fInput[i, 0] > fMax)
                    fMax = fInput[i, 0];
                if (fInput[i, 0] < fMin)
                    fMin = fInput[i, 0];
                //s
                if (fInput[i, 1] > sMax)
                    sMax = fInput[i, 1];
                if (fInput[i, 1] < sMin)
                    sMin = fInput[i, 1];
                //t
                if (fInput[i, 2] > tMax)
                    tMax = fInput[i, 2];
                if (fInput[i, 2] < tMin)
                    tMin = fInput[i, 2];
                //d
                if (fInput[i, 3] > dMax)
                    dMax = fInput[i, 3];
                if (fInput[i, 3] < dMin)
                    dMin = fInput[i, 3];
            }

            System.Console.WriteLine("Start training--- input Normalization");
            //make input between 0 ~ 1
            for (int i = 0; i < sampleNum; i++)
            {
                fInput[i, 0] = (fInput[i, 0] - fMin) / (fMax - fMin);
                fInput[i, 1] = (fInput[i, 1] - sMin) / (sMax - sMin);
                fInput[i, 2] = (fInput[i, 2] - tMin) / (tMax - tMin);
                fInput[i, 3] = (fInput[i, 3] - dMin) / (dMax - dMin);
            }

            System.Console.WriteLine("Start training---learnning process");
            for (int sampleID = 0; sampleID < sampleNum; sampleID++)
            {
                //get the stand output, first 50 is  1, rest is 0
                if (sampleID <= 50)
                {
                    standOut = 1;
                }

                else
                {
                    standOut = 0;
                }

                //initial the hide_input and hide_out
                for (int i = 0; i < hide_Num; i++)
                {
                    input_hide[i] = 0.0;
                    output_hide[i] = 0.0;
                    inputError[i] = 0.0;
                }

                //initial the output_input and output_output
                for (int i = 0; i < output_Num; i++)
                {
                    input_output[i] = 0.0;
                    output_output[i] = 0.0;
                }

                //System.Console.WriteLine("Start training---start calculate numbers");
                //---------------------------------------------------------------------
                //calculate output of hidden layer
                //---------------------------------------------------------------------
                for (int i = 0; i < hide_Num; i++)
                {
                    //get input of hidden node(input * weight)
                    for (int j = 0; j < input_Num; j++)
                    {
                        input_hide[i] = input_hide[i] + (fInput[sampleID, j] * weight_hidden[j, i]);
                    }
                    //calculate output
                    output_hide[i] = 1.0 / (1.0 + Math.Exp(-1.0 * input_hide[i]));
                }

                //---------------------------------------------------------------------
                //calculate output of the output layer
                //---------------------------------------------------------------------
                for (int i = 0; i < output_Num; i++)
                {
                    for (int j = 0; j < hide_Num; j++)
                    {
                        input_output[i] = input_output[i] + (output_hide[j] * weight_output[j, i]);
                    }
                    output_output[i]
                        = 1.0 / (1.0 + Math.Exp(-1.0 * input_output[i]));

                    for(int n = 0; n <hide_Num;n++)
                    {
                        //find the rate to fix the weight, use it later
                        rate[n, i] = -1 * (standOut - output_output[i]) * output_output[i] * (1 - output_output[i]) * output_output[i];
                    }

                }
                //calculate final learning error rate
                if (output_output[0] >= 0.5)
                {
                    if (standOut == 1)
                        rightNum++;
                    else
                        wrongNum++;
                }
                else
                {
                    if (standOut == 0)
                        rightNum++;
                    else
                        wrongNum++;
                }

                //start a new calculate for input-->hidden
                for (int i = 0; i < hide_Num; i++)
                {
                    for (int j = 0; j < output_Num; j++)
                    {
                        //cause I need old weight here, so have to fix weight later hidden-->output
                        inputError[i] = inputError[i] +(-1 * (standOut - output_output[j] * output_output[j] * (1 - output_output[j]) * weight_hidden[i,j]));
                        //now I don't need old weight anymore, so fix the weight hidden-->output
                        weight_output[i, j] = weight_output[i, j] - (0.5 * rate[i, j]);
                    }

                    for(int k = 0;k<input_Num;k++)
                    {
                        //get the rate for fix the weight intput-->hidden
                        inputrate[k, i] = inputError[i] * output_hide[i] * (1 - output_hide[i]) * weight_hidden[k, i];
                        //fix weight input-->hidden
                        weight_hidden[k, i] = weight_hidden[k, i] - (0.5 * inputrate[k,i]);
                    }
                }
                //----------------------------------------------------------------------
                //=---------------------------------------------------------------------
                //READYFOR BACKWORDS
                //----------------------------------------------------------------------
                //----------------------------------------------------------------------

                System.Console.WriteLine("---------------------------------------------------------------------------------------------------------------------");
                System.Console.WriteLine("---------------------------------------------------------------------------------------------------------------------");
                System.Console.WriteLine("output_output is {0}", output_output[0]);
                System.Console.WriteLine("target output is {0}", standOut);
                /*System.Console.WriteLine("---------------------------------------------------------------------------------------------------------------------");
                System.Console.WriteLine("weight-input  {0}---{1}---{2}:", weight_output[0, 0], weight_output[1, 0], weight_output[2, 0],weight_output[3,0]);
                System.Console.WriteLine("---------------------------------------------------------------------------------------------------------------------");
                System.Console.WriteLine("weight-hidden  {0} {1} {2} {3}:", weight_hidden[0, 0], weight_hidden[0, 1], weight_hidden[0, 2], weight_hidden[0, 3]);
                System.Console.WriteLine("weight-hidden  {0} {1} {2} {3}:", weight_hidden[1, 0], weight_hidden[1, 1], weight_hidden[1, 2], weight_hidden[1, 3]);
                System.Console.WriteLine("weight-hidden  {0} {1} {2} {3}:", weight_hidden[2, 0], weight_hidden[2, 1], weight_hidden[2, 2], weight_hidden[2, 3]);
                System.Console.WriteLine("weight-hidden  {0} {1} {2} {3}:", weight_hidden[3, 0], weight_hidden[3, 1], weight_hidden[3, 2], weight_hidden[3, 3]);
                //-------------------------------------------------------------------------------------------------------------*/
                Thread.Sleep(10);
            }
            System.Console.WriteLine("Start training---learning done");

        }

        public int test(string input)
        {
            string[] data = new string[4];
            double[] intData = new double[4];

            data = input.Split(',');

            intData[0] = Convert.ToDouble(data[0]);
            intData[1] = Convert.ToDouble(data[1]);
            intData[2] = Convert.ToDouble(data[2]);
            intData[3] = Convert.ToDouble(data[3]);


            //calculate output
            for (int i = 0; i < hide_Num; i++)
            {
                //get input of hidden node(input * weight)
                for (int j = 0; j < input_Num; j++)
                {
                    input_hide[i] = input_hide[i] + (intData[j] * weight_hidden[j, i]);
                }
                //calculate output
                output_hide[i] = 1.0 / (1.0 + Math.Exp(-1.0 * input_hide[i]));
            }

            for (int i = 0; i < output_Num; i++)
            {
                for (int j = 0; j < hide_Num; j++)
                {
                    input_output[i] = input_output[i] + (output_hide[j] * weight_output[j, i]);
                }
                output_output[i] = 1.0 / (1.0 + Math.Exp(-1.0 * input_output[i]));
            }

            if (output_output[0] >= 0.5)
                return 1;
            else
                return 0;
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            string[] readData;
            double[,] inputData = new double[151, 4];
            double[,] testData = new double[151,4];
            int studyCounter = 0;
            /**
            ***  read dataset into array
            **/
            using (System.IO.StreamReader sr = new System.IO.StreamReader(@"C:\Data_Set.txt")) // hardcode path, should be user input in formual
            {
                string str;
                int sampleCounter = 1;
                while ((str = sr.ReadLine()) != null)
                {
                    readData = str.Split(',');
                    
                    inputData[sampleCounter, 0] = Convert.ToDouble(readData[0]);
                    inputData[sampleCounter, 1] = Convert.ToDouble(readData[1]);
                    inputData[sampleCounter, 2] = Convert.ToDouble(readData[2]);
                    inputData[sampleCounter, 3] = Convert.ToDouble(readData[3]);
                    System.Console.WriteLine("read done: {0}",sampleCounter);
                    sampleCounter++;
                }
            }

            //create network
            NNet nerual = new NNet();

            //train 
            do
            {
                nerual.train(inputData);
                studyCounter++;
                System.Console.WriteLine("Error percent: {0} %",(decimal)nerual.wrongNum/(nerual.rightNum + nerual.wrongNum));
            } while (studyCounter < 5);

            System.Console.ReadLine();

            /*//test
            using (System.IO.StreamReader sr1 = new System.IO.StreamReader(@"C:\Data_Set.txt"))
            {
                string str;
                int testCounter = 0;
                while ((str = sr1.ReadLine()) != null)
                {
                    int result = nerual.test(str);

                    if(testCounter<50)
                        System.Console.WriteLine("read done: {0}-- 1 ", result);
                    else
                        System.Console.WriteLine("read done: {0}-- 0 ", result);

                    testCounter++;
                }
            }*/

           

            //cout

            System.Console.ReadLine();
        }
    }
}
//end of namespace











