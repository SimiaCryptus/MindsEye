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

package com.simiacryptus.mindseye.net.util;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

import java.util.List;

public class VariableLayer extends NNLayer {

  private static final long serialVersionUID = 6284058717982209085L;
  private NNLayer inner;

  public VariableLayer(final NNLayer inner) {
    super();
    setInner(inner);
  }

  @Override
  public NNResult eval(final NNResult... array) {
    return getInner().eval(array);
  }

  @Override
  public List<NNLayer> getChildren() {
    return super.getChildren();
  }

  public final NNLayer getInner() {
    return this.inner;
  }

  public final void setInner(final NNLayer inner) {
    this.inner = inner;
  }

  @Override
  public JsonObject getJson() {
    return this.inner.getJson();
  }

  @Override
  public List<double[]> state() {
    return getInner().state();
  }

}
