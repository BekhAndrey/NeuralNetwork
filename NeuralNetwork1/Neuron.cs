using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }


        public Neuron(int inputAmount, NeuronType type = NeuronType.Normal)
        {
            NType = type;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitWeightsRandomValue(inputAmount);

        }

        private void InitWeightsRandomValue(int inputAmount)
        {
            var rnd = new Random();
            for (int i = 0; i < inputAmount; i++)
            {
                if(NType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> input)
        {
            for (int i = 0; i < input.Count; i++)
            {
                Inputs[i] = input[i];
            }
            var sum = 0.0;
            for (int i = 0; i < input.Count; i++)
            {
                sum += input[i] * Weights[i];
            }
            if (NType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }
            return Output;
        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Exp(-x));
            return result;
        }

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid * (1 - sigmoid);
            return result;
        }

        public void Learn(double error, double learningRate)
        {
            if (NType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
