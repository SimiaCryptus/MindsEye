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
    setFrozen(true);
  }
  
  /**
   * Instantiates a new Simple activation layer.
   *
   * @param id the id
   */
  protected SimpleActivationLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * Eval.
   *
   * @param x       the x
   * @param results the results
   */
  protected abstract void eval(final double x, double[] results);
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    assert 0 < itemCnt;
    final Tensor inputGradientA[] = new Tensor[itemCnt];
    final Tensor[] outputA = IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      final Tensor output = new Tensor(inObj[0].getData().get(dataIndex).getDimensions());
      final Tensor inputGradient = new Tensor(input.dim());
      inputGradientA[dataIndex] = inputGradient;
      final double[] results = new double[2];
      for (int i = 0; i < input.dim(); i++) {
        eval(input.getData()[i], results);
        inputGradient.set(i, results[1]);
        output.set(i, results[0]);
      }
      return output;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
  
      @Override
      public void free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (inObj[0].isAlive()) {
          final Tensor[] passbackA = IntStream.range(0, itemCnt).parallel().mapToObj(dataIndex -> {
            final Tensor passback = new Tensor(data.get(dataIndex).getDimensions());
            final double[] gradientData = inputGradientA[dataIndex].getData();
            IntStream.range(0, passback.dim()).forEach(i -> {
              final double v = gradientData[i];
              if (Double.isFinite(v)) {
                passback.set(i, data.get(dataIndex).getData()[i] * v);
              }
            });
            return passback;
          }).toArray(i -> new Tensor[i]);
          inObj[0].accumulate(buffer, new TensorArray(passbackA));
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
