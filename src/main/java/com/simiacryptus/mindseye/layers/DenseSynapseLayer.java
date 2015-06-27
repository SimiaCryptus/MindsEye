package com.simiacryptus.mindseye.layers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.FeedbackContext;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.NNResult;

public class DenseSynapseLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);
  
  private final int[] outputDims;
  public final NDArray weights;
  private boolean frozen = false;
  private NDArray bufferGradient;
  private double[] bufferFeedback;
  private List<NNResult> predecessors = new ArrayList<NNResult>();
  private int bufferPos = 0;
  
  public DenseSynapseLayer(int inputs, int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
  }
  
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final int[] inputDims = input.getDims();
    final NDArray output = new NDArray(outputDims);
    final NDArray inputGradient = new NDArray(input.dim(), output.dim());
    final NDArray weightGradient = this.frozen?null:new NDArray(weights.dim(), output.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      IntStream.range(0, output.dim()).forEach(o -> {
        double a = weights.get(i, o);
        double b = input.data[i];
        inputGradient.add(new int[] { i, o }, a);
        if(null!=weightGradient) weightGradient.add(new int[] { weights.index(i, o), o }, b);
        output.add(o, b * a);
      });
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data, FeedbackContext ctx) {
        if(null!=weightGradient) {
          double[] invertFeedback = ctx.invertFeedback(weightGradient, data.data);
          ctx.adjust(DenseSynapseLayer.this, weights, invertFeedback);
        }
        synchronized (DenseSynapseLayer.this) {
          if (inObj.isAlive()) {
            
            if (0 == bufferPos) {
              int inx = input.dim();
              int outx = output.dim();
              int endx = output.dim();
              while (endx < inx)
                endx += outx;
              bufferGradient = new NDArray(inx, endx);
              bufferFeedback = new double[endx];
            }
            for (int i = 0; i < output.dim(); i++)
            {
              for (int j = 0; j < input.dim(); j++)
              {
                bufferGradient.set(new int[] { j, bufferPos }, inputGradient.get(j, i));
              }
              bufferFeedback[bufferPos] = data.data[i];
              bufferPos++;
            }
            predecessors.add(inObj);
            if (bufferPos >= bufferGradient.getDims()[1]) {
              double[] inverted = ctx.invertFeedback(bufferGradient, bufferFeedback);
              bufferPos=0;
              for (NNResult predecessor : predecessors) {
                double[] feedbackChunk = new double[predecessor.data.dim()];
                for (int i = 0; i < output.dim(); i++)
                {
                  feedbackChunk[i] = inverted[bufferPos++];
                }
                predecessor.feedback(new NDArray(predecessor.data.getDims(), feedbackChunk));
              }
              bufferPos=0;
              predecessors.clear();
              //inObj.feedback(inverted, ctx);
            }
          }
        }
        
      }
      
      public boolean isAlive() {
        return null!=weightGradient||inObj.isAlive();
      }
    };
  }

  public DenseSynapseLayer addWeights(DoubleSupplier f) {
    for(int i=0;i<weights.data.length;i++)
    {
      weights.data[i] += f.getAsDouble();
    }
    return this;
  }

  public DenseSynapseLayer setWeights(DoubleSupplier f) {
    Arrays.parallelSetAll(weights.data, i->f.getAsDouble());
    return this;
  }

  public DenseSynapseLayer freeze() {
    return freeze(true);
  }

  public DenseSynapseLayer thaw() {
    return freeze(false);
  }

  public DenseSynapseLayer freeze(boolean b) {
    this.frozen = b;
    return this;
  }
  
}
