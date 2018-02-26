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

package com.simiacryptus.mindseye.lang;

import com.simiacryptus.mindseye.layers.java.PlaceholderLayer;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.function.BiConsumer;

public class MutableResult extends Result {
  
  public MutableResult(final Tensor... tensors) {
    super(TensorArray.create(tensors), handler(tensors));
  }
  
  private static BiConsumer<DeltaSet<Layer>, TensorList> handler(final Tensor[] tensors) {
    return (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      for (int index = 0; index < delta.length(); index++) {
        final Tensor dt = delta.get(index);
        @Nullable final double[] p = tensors[index].getData();
        @Nonnull PlaceholderLayer<double[]> layer = new PlaceholderLayer<>(p);
        buffer.get(layer, p).addInPlace(dt.getData()).freeRef();
        dt.freeRef();
        layer.freeRef();
      }
    };
  }
  
  @Override
  public boolean isAlive() {
    return true;
  }
}
