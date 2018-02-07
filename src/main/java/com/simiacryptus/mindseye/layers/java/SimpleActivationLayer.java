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
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * A parent class for all stateless, univariate "activation" functions.
 *
 * @param <T> the type parameter
 */
@SuppressWarnings("serial")
public abstract class SimpleActivationLayer<T extends SimpleActivationLayer<T>> extends NNLayer {
  
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
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    assert 0 < itemCnt;
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    @javax.annotation.Nonnull final Tensor inputGradientA[] = new Tensor[itemCnt];
    return new NNResult(TensorArray.wrap(IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      @javax.annotation.Nonnull final Tensor output = new Tensor(inObj[0].getData().getDimensions());
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
    }).toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      if (inObj[0].isAlive()) {
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
          @javax.annotation.Nonnull final Tensor passback = new Tensor(data.getDimensions());
          final @Nullable double[] gradientData = inputGradientA[dataIndex].getData();
          IntStream.range(0, passback.dim()).forEach(i -> {
            final double v = gradientData[i];
            if (Double.isFinite(v)) {
              passback.set(i, data.get(dataIndex).getData()[i] * v);
            }
          });
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
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
