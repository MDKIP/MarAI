using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MarAI.Visualization
{
    public interface IDirectParametersVisualizer
    {
        int ImageWidth { get; }
        int ImageHeight { get; }
        DirectParametersVisualizationType VisualizationType { get; }

        Bitmap Visualize();
    }
}
