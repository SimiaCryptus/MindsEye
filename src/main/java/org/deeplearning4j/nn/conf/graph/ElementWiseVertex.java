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

package org.deeplearning4j.nn.conf.graph;

import org.deeplearning4j.nn.conf.inputs.InputType;

public class ElementWiseVertex implements GraphVertex {
  public ElementWiseVertex(Op mergeMode) {
  
  }
  
  @Override
  public InputType getOutputType(int i, InputType[] inputType) {
    throw new RuntimeException("NI");
  }
  
  public enum Op {
    Product, Add
  }
}
