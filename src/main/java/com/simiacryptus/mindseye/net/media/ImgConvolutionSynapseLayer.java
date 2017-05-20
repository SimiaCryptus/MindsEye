/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.net.media;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.opencl.ConvolutionController;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

public class ImgConvolutionSynapseLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(ImgConvolutionSynapseLayer.class);
  public static final Function<IndexMapKey, ConvolutionController> cache = Util.cache((final IndexMapKey key) -> {
    synchronized (ImgConvolutionSynapseLayer.class) {
      final int outDim = new Tensor(key.output).dim();
      final int inDim = new Tensor(key.input).dim();
      log.debug(String.format("%s ins * %s bands => %s outs", inDim, Arrays.toString(key.kernel), outDim));
      assert 3 == key.input.length;
      assert 3 == key.kernel.length;
      final ConvolutionController kernels = new ConvolutionController(key.input, key.kernel);
      log.debug("Commputed kernels for " + key + ": " + kernels);
      return kernels;
    }
    
  });
  private static final long serialVersionUID = -139062498597441290L;
  public final Tensor kernel;
  
  protected ImgConvolutionSynapseLayer() {
    this(null);
  }
  
  protected ImgConvolutionSynapseLayer(Tensor kernel) {
    super();
    if(kernel.getDims().length != 3) throw new IllegalArgumentException();
    if(kernel.getDims()[0] % 2 != 1) throw new IllegalArgumentException();
    if(kernel.getDims()[1] % 2 != 1) throw new IllegalArgumentException();
    this.kernel = kernel;
  }
  
  public ImgConvolutionSynapseLayer(final int width, int height, final int inputBands, final int outputBands) {
    this(width, height, inputBands * outputBands);
  }
  
  public ImgConvolutionSynapseLayer(final int width, int height, final int bands) {
    this(new Tensor(width,height,bands));
  }
  
  public static int[] getOutputDims(final int[] inputSize, final int[] kernelSize) {
    return IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      if (i == kernelSize.length - 1) {
        x = kernelSize[i] / inputSize[i];
      } else {
        x = inputSize[i];
      }
      if (0 >= x) {
        assert false;
      }
      return x;
    }).toArray();
  }
  
  public ImgConvolutionSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    final ConvolutionController convolutionController;
    final NNResult input = inObj[0];
    final Tensor[] batch = input.data;
    final int[] inputDims = batch[0].getDims();
    {
      final int[] kernelDims = this.kernel.getDims();
      convolutionController = cache.apply(new IndexMapKey(kernelDims, inputDims, getOutputDims(inputDims, kernelDims)));
    }
    int[] outputDims = getOutputDims(inputDims, this.kernel.getDims());
    Tensor[] outputA = IntStream.range(0, batch.length).mapToObj(dataIndex -> new Tensor(outputDims)).toArray(i -> new Tensor[i]);
    {
      double[][] inputBuffers = Arrays.stream(batch).map(x -> x.getData()).toArray(i -> new double[i][]);
      double[][] outputBuffers = Arrays.stream(outputA).map(x -> x.getData()).toArray(i -> new double[i][]);
      convolutionController.convolve(inputBuffers, this.kernel.getData(), outputBuffers);
    }
    
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] error) {
        if (!isFrozen()) {
          double[][] inputBuffers = Arrays.stream(batch).map(x -> x.getData()).toArray(i -> new double[i][]);
          double[][] outputBuffers = Arrays.stream(error).map(x -> x.getData()).toArray(i -> new double[i][]);
          final Tensor kernel = ImgConvolutionSynapseLayer.this.kernel;
          final Tensor weightGradient = new Tensor(kernel.getDims());
          convolutionController.gradient(inputBuffers, weightGradient.getData(), outputBuffers);
          buffer.get(ImgConvolutionSynapseLayer.this, kernel).accumulate(weightGradient.getData());
        }
        if (input.isAlive()) {
          Tensor[] inputBufferTensors = IntStream.range(0, data.length).mapToObj(dataIndex -> new Tensor(inputDims)).toArray(i -> new Tensor[i]);
          double[][] inputBuffers = Arrays.stream(inputBufferTensors).map(x -> x.getData()).toArray(i -> new double[i][]);
          double[][] outputBuffers = Arrays.stream(error).map(x -> x.getData()).toArray(i -> new double[i][]);
          convolutionController.backprop(inputBuffers, ImgConvolutionSynapseLayer.this.kernel.getData(), outputBuffers);
          input.accumulate(buffer, inputBufferTensors);
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("kernel", this.kernel.toString());
    return json;
  }
  
  public ImgConvolutionSynapseLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.kernel.coordStream().parallel().forEach(c -> {
      this.kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
  }
  
  public static final class IndexMapKey {
    int[] input;
    int[] kernel;
    int[] output;
    
    public IndexMapKey(final int[] kernel, final int[] input, final int[] output) {
      super();
      this.kernel = kernel;
      this.input = input;
      this.output = output;
      assert 3 == input.length;
      assert 3 == kernel.length;
    }
    
    @Override
    public boolean equals(final Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      final IndexMapKey other = (IndexMapKey) obj;
      if (!Arrays.equals(this.input, other.input))
        return false;
      if (!Arrays.equals(this.kernel, other.kernel))
        return false;
      return Arrays.equals(this.output, other.output);
    }
    
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(this.input);
      result = prime * result + Arrays.hashCode(this.kernel);
      result = prime * result + Arrays.hashCode(this.output);
      return result;
    }
    
    @Override
    public String toString() {
      final StringBuilder builder = new StringBuilder();
      builder.append("IndexMapKey [input=");
      builder.append(Arrays.toString(this.input));
      builder.append(", kernel=");
      builder.append(Arrays.toString(this.kernel));
      builder.append(", output=");
      builder.append(Arrays.toString(this.output));
      builder.append("]");
      return builder.toString();
    }
    
  }
}
