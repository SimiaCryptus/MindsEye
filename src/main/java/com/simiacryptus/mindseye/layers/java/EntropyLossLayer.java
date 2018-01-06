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
import java.util.Map;
import java.util.stream.IntStream;

/**
 * An entropy-based cost function. The output value is the expected number of nats needed to encode a category chosen
 * using the first input as a distribution, but using the second input distribution for the encoding scheme.
 */
@SuppressWarnings("serial")
public class EntropyLossLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);
  
  /**
   * Instantiates a new Entropy loss layer.
   */
  public EntropyLossLayer() {
  }
  
  /**
   * Instantiates a new Entropy loss layer.
   *
   * @param id the id
   */
  protected EntropyLossLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json entropy loss layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the entropy loss layer
   */
  public static EntropyLossLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new EntropyLossLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final double zero_tol = 1e-12;
    final Tensor gradient[] = new Tensor[inObj[0].getData().length()];
    final double max_prob = 1.;
    final Tensor[] output = IntStream.range(0, inObj[0].getData().length()).mapToObj(dataIndex -> {
      final Tensor l = inObj[0].getData().get(dataIndex);
      final Tensor r = inObj[1].getData().get(dataIndex);
      assert l.dim() == r.dim() : l.dim() + " != " + r.dim();
      final Tensor gradientTensor = new Tensor(l.getDimensions());
      final double[] gradientData = gradientTensor.getData();
      double total = 0;
      final double[] ld = l.getData();
      final double[] rd = r.getData();
      for (int i = 0; i < l.dim(); i++) {
        final double lv = Math.max(Math.min(ld[i], max_prob), zero_tol);
        final double rv = rd[i];
        if (rv > 0) {
          gradientData[i] = -rv / lv;
          total += -rv * Math.log(lv);
        }
        else {
          gradientData[i] = 0;
        }
      }
      assert total >= 0;
      gradient[dataIndex] = gradientTensor;
      final Tensor outValue = new Tensor(new double[]{total}, 1);
      return outValue;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(output) {
  
      @Override
      public void free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (inObj[1].isAlive()) {
          inObj[1].accumulate(buffer, new TensorArray(IntStream.range(0, data.length()).mapToObj(dataIndex -> {
            final Tensor l = inObj[0].getData().get(dataIndex);
            final Tensor passback = new Tensor(gradient[dataIndex].getDimensions());
            for (int i = 0; i < passback.dim(); i++) {
              final double lv = Math.max(Math.min(l.get(i), max_prob), zero_tol);
              passback.set(i, -data.get(dataIndex).get(0) * Math.log(lv));
            }
            return passback;
          }).toArray(i -> new Tensor[i])));
        }
        if (inObj[0].isAlive()) {
          inObj[0].accumulate(buffer, new TensorArray(IntStream.range(0, data.length()).mapToObj(dataIndex -> {
            final Tensor passback = new Tensor(gradient[dataIndex].getDimensions());
            for (int i = 0; i < passback.dim(); i++) {
              passback.set(i, data.get(dataIndex).get(0) * gradient[dataIndex].get(i));
            }
            return passback;
          }).toArray(i -> new Tensor[i])));
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || inObj[0].isAlive();
      }
      
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
