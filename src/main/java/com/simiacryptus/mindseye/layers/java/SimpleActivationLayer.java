/*
 * Copyright (c) 2018 by Andrew Charneski.
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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * A parent class for all stateless, univariate "activation" functions.
 *
 * @param <T> the type parameter
 */
@SuppressWarnings("serial")
public abstract class SimpleActivationLayer<T extends SimpleActivationLayer<T>> extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);
  
  /**
   * Instantiates a new Simple activation layer.
   */
  public SimpleActivationLayer() {
    super();
    this.frozen = true;
  }
  
  /**
   * Instantiates a new Simple activation layer.
   *
   * @param id the id
   */
  protected SimpleActivationLayer(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * Eval.
   *
   * @param x       the x
   * @param results the results
   */
  protected abstract void eval(final double x, double[] results);
  
  @javax.annotation.Nonnull
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    final TensorList indata0 = inObj[0].getData();
    final int itemCnt = indata0.length();
    assert 0 < itemCnt;
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    Arrays.stream(inObj).forEach(nnResult -> nnResult.getData().addRef());
    @javax.annotation.Nonnull final Tensor inputGradientA[] = new Tensor[itemCnt];
    return new Result(TensorArray.wrap(IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
      @javax.annotation.Nullable final Tensor input = indata0.get(dataIndex);
      @javax.annotation.Nonnull final Tensor output = new Tensor(indata0.getDimensions());
      @javax.annotation.Nonnull final Tensor inputGradient = new Tensor(input.dim());
      inputGradientA[dataIndex] = inputGradient;
      @javax.annotation.Nonnull final double[] results = new double[2];
      for (int i = 0; i < input.dim(); i++) {
        eval(input.getData()[i], results);
        inputGradient.set(i, results[1]);
        output.set(i, results[0]);
      }
      input.freeRef();
      return output;
    }).toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      if (inObj[0].isAlive()) {
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
          @javax.annotation.Nonnull final Tensor passback = new Tensor(data.getDimensions());
          @Nullable final double[] gradientData = inputGradientA[dataIndex].getData();
          @javax.annotation.Nullable Tensor tensor = data.get(dataIndex);
          IntStream.range(0, passback.dim()).forEach(i -> {
            final double v = gradientData[i];
            if (Double.isFinite(v)) {
              passback.set(i, tensor.get(i) * v);
            }
          });
          tensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        Arrays.stream(inObj).forEach(nnResult -> nnResult.getData().freeRef());
        for (@javax.annotation.Nonnull Tensor tensor : inputGradientA) {
          tensor.freeRef();
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
