using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MarAI.NeuralNetworks;
using MarAI.Mathematics;

namespace MarAI.Visualization
{
    public class PerceptronDirectParametersVisualizer : IDirectParametersVisualizer
    {
        public PerceptronDirectParametersVisualizer(IPerceptron perceptron, DirectParametersVisualizationType visualizationType)
        {
            VisualizationType = visualizationType;
            Unit = perceptron;
            
            switch (VisualizationType)
            {
                case DirectParametersVisualizationType.SquareOfRGB:
                    if (Math.Sqrt(Unit.InputsNumber / 3) % 1 != 0)
                    {
                        throw new ArgumentException("Visualization Impossible because of wrong size", "perceptron");
                    }
                    ImageWidth = (int)Math.Sqrt(Unit.InputsNumber / 3);
                    ImageHeight = (int)Math.Sqrt(Unit.InputsNumber / 3);
                    break;
                case DirectParametersVisualizationType.SquareOfGray:
                    if (Math.Sqrt(Unit.InputsNumber) % 1 != 0)
                    {
                        throw new ArgumentException("Visualization Impossible because of wrong size", "perceptron");
                    }
                    ImageWidth = (int)Math.Sqrt(Unit.InputsNumber);
                    ImageHeight = (int)Math.Sqrt(Unit.InputsNumber);
                    break;
                case DirectParametersVisualizationType.BeltOfRGB:
                    if ((Unit.InputsNumber / 3) % 1 != 0)
                    {
                        throw new ArgumentException("Visualization Impossible because of wrong size", "perceptron");
                    }
                    ImageWidth = 1;
                    ImageHeight = (int)Unit.InputsNumber / 3;
                    break;
                case DirectParametersVisualizationType.BeltOfGray:
                    ImageWidth = 1;
                    ImageHeight = (int)Unit.InputsNumber;
                    break;
            }
        }

        public IPerceptron Unit { get; private set; }
        public int ImageWidth { get; private set; }
        public int ImageHeight { get; private set; }
        public DirectParametersVisualizationType VisualizationType { get; private set; }

        public Bitmap Visualize()
        {
            Bitmap img = new Bitmap(ImageWidth, ImageHeight);

            decimal min = decimal.MaxValue;
            decimal max = decimal.MinValue;
            for (int y = 0; y < Unit.Weights.Length; y++)
            {
                if (Unit.Weights[y] > max)
                {
                    max = Unit.Weights[y];
                }
                if (Unit.Weights[y] < min)
                {
                    min = Unit.Weights[y];
                }
            }

            switch (VisualizationType)
            {
                case DirectParametersVisualizationType.BeltOfGray:
                    for (int y = 0; y < ImageHeight; y++)
                    {
                        int g = (int)Unit.Weights[y].Map(min, max, 0, 255);
                        img.SetPixel(0, y, Color.FromArgb(g, g, g));
                    }
                    break;
                case DirectParametersVisualizationType.BeltOfRGB:
                    for (int y = 0; y < ImageHeight * 3; y += 3)
                    {
                        int r = (int)Unit.Weights[y+0].Map(min, max, 0, 255);
                        int g = (int)Unit.Weights[y+1].Map(min, max, 0, 255);
                        int b = (int)Unit.Weights[y+2].Map(min, max, 0, 255);
                        img.SetPixel(0, y / 3, Color.FromArgb(r, g, b));
                    }
                    break;
                case DirectParametersVisualizationType.SquareOfGray:
                    for (int y = 0; y < ImageHeight; y++)
                    {
                        for (int x = 0; x < ImageWidth; x++)
                        {
                            int g = (int)Unit.Weights[y * ImageWidth + x].Map(min, max, 0, 255);
                            img.SetPixel(x, y, Color.FromArgb(g, g, g));
                        }
                    }
                    break;
                case DirectParametersVisualizationType.SquareOfRGB:
                    for (int y = 0; y < ImageHeight * 3; y += 3)
                    {
                        for (int x = 0; x < ImageWidth * 3; x += 3)
                        {
                            int r = (int)Unit.Weights[y * ImageWidth + x + 0].Map(min, max, 0, 255);
                            int g = (int)Unit.Weights[y * ImageWidth + x + 1].Map(min, max, 0, 255);
                            int b = (int)Unit.Weights[y * ImageWidth + x + 2].Map(min, max, 0, 255);
                            img.SetPixel(x / 3, y / 3, Color.FromArgb(r, g, b));
                        }
                    }
                    break;
            }

            return img;
        }
    }
}
