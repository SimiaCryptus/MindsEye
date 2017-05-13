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

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@SuppressWarnings("serial")
public final class VerboseWrapper extends NNLayer {
  
  static final Logger log = LoggerFactory.getLogger(VerboseWrapper.class);
  
  public final NNLayer inner;
  public final String label;
  
  public VerboseWrapper(final String label, final NNLayer inner) {
    this.inner = inner;
    this.label = label;
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    final NNResult result = this.inner.eval(inObj);
    log.debug(String.format("%s: %s => %s", this.label,
        Arrays.stream(inObj).map(l -> Arrays.toString(l.data)).collect(Collectors.toList()),
        Arrays.toString(result.data)));
    return result;
  }
  
  @Override
  public List<double[]> state() {
    return this.inner.state();
  }
}
