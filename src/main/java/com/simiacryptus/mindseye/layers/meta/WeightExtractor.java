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

package com.simiacryptus.mindseye.layers.meta;

import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

@SuppressWarnings("serial")
public final class WeightExtractor extends NNLayer {

  static final Logger log = LoggerFactory.getLogger(WeightExtractor.class);

  private final NNLayer inner;
  private final int index;

  public WeightExtractor(final int index, final NNLayer inner) {
    this.inner = inner;
    this.index = index;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    Tensor array = new Tensor(inner.state().get(index));
    return new NNResult(array) {
      
      @Override
      public boolean isAlive() {
        return true;
      }
      
      @Override
      public void accumulate(DeltaSet buffer, Tensor[] data) {
        assert (data.length == 1);
        buffer.get(WeightExtractor.this, array).accumulate(data[0].getData());
      }
    };
  }

  @Override
  public List<double[]> state() {
    return this.inner.state();
  }
}
