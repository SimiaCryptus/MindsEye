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

package com.simiacryptus.mindseye.test.integration;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.util.test.LabeledObject;

import java.io.IOException;
import java.util.stream.Stream;

/**
 * The type Mnist problem data.
 */
public class MnistProblemData implements ImageProblemData {
  
  @Override
  public Stream<LabeledObject<Tensor>> trainingData() throws IOException {
    System.out.println(String.format("Loaded %d items", MNIST.trainingDataStream().count()));
    return MNIST.trainingDataStream();
  }
  
  @Override
  public Stream<LabeledObject<Tensor>> validationData() throws IOException {
    return MNIST.validationDataStream();
  }
  
}
