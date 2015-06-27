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
  
  
  private NDArray bufferWeightGradient;
  private double[] bufferWeightSignal;
  private int bufferWeightPos = 0;
  
  private NDArray bufferFeedbackGradient;
  private double[] bufferFeedbackSignal;
  private List<NNResult> bufferFeedbackNext = new ArrayList<NNResult>();
  private int bufferFeedbackPos = 0;
  
  
  
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
        synchronized (DenseSynapseLayer.this) {
          if(null!=weightGradient) {
            if (0 == bufferWeightPos) {
              int inx = weights.dim();
              int outx = output.dim();
              int endx = output.dim();
              while (endx < inx)
                endx += outx;
              bufferWeightGradient = new NDArray(inx, endx);
              bufferWeightSignal = new double[endx];
            }
            for (int i = 0; i < output.dim(); i++)
            {
              for (int j = 0; j < weights.dim(); j++)
              {
                bufferWeightGradient.set(new int[] { j, bufferWeightPos }, weightGradient.get(j, i));
              }
              bufferWeightSignal[bufferWeightPos] = data.data[i];
              bufferWeightPos++;
            }
            if (bufferWeightPos >= bufferWeightGradient.getDims()[1]) {
              double[] inverted = ctx.invertFeedback(bufferWeightGradient, bufferWeightSignal);
              bufferWeightPos=0;
              while(bufferWeightPos < bufferWeightGradient.getDims()[1]) {
                double[] chunk = new double[weights.dim()];
                for (int i = 0; i < chunk.length; i++)
                {
                  chunk[i] = inverted[bufferWeightPos++];
                }
                ctx.adjust(DenseSynapseLayer.this, weights, chunk);
              }
              bufferWeightPos=0;
            }
          }
          if (inObj.isAlive()) {
            if (0 == bufferFeedbackPos) {
              int inx = input.dim();
              int outx = output.dim();
              int endx = output.dim();
              while (endx < inx)
                endx += outx;
              bufferFeedbackGradient = new NDArray(inx, endx);
              bufferFeedbackSignal = new double[endx];
            }
            assert(bufferFeedbackPos < bufferFeedbackGradient.getDims()[1]);
            for (int i = 0; i < output.dim(); i++)
            {
              for (int j = 0; j < input.dim(); j++)
              {
                bufferFeedbackGradient.set(new int[] { j, bufferFeedbackPos }, inputGradient.get(j, i));
              }
              bufferFeedbackSignal[bufferFeedbackPos] = data.data[i];
              bufferFeedbackPos++;
            }
            assert(bufferFeedbackPos > 0);
            bufferFeedbackNext.add(inObj);
            if (bufferFeedbackPos >= bufferFeedbackGradient.getDims()[1]) {
              double[] inverted = ctx.invertFeedback(bufferFeedbackGradient, bufferFeedbackSignal);
              assert(inverted.length == bufferFeedbackGradient.getDims()[0]);
              bufferFeedbackPos=0;
              while(bufferFeedbackPos < inverted.length) {
                NNResult predecessor = bufferFeedbackNext.remove(0);
                double[] chunk = new double[predecessor.data.dim()];
                for (int i = 0; i < chunk.length; i++)
                {
                  chunk[i] = inverted[bufferFeedbackPos++];
                }
                predecessor.feedback(new NDArray(predecessor.data.getDims(), chunk));
              }
              bufferFeedbackPos=0;
              bufferFeedbackNext.clear();
            }
            assert(bufferFeedbackPos < bufferFeedbackGradient.getDims()[0]);
            assert(bufferFeedbackPos >= 0);
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
