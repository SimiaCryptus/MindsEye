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
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.test.LabeledObject;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Caltech 101 Image Dataset.
 */
public class CaltechProblemData implements ImageProblemData {
  
  private final int imageSize;
  private List<String> labels = null;
  
  public CaltechProblemData() {
    this(256);
  }
  
  public CaltechProblemData(int imageSize) {
    this.imageSize = imageSize;
  }
  
  @Override
  public Stream<LabeledObject<Tensor>> trainingData() {
    try {
      return Caltech101.trainingDataStream().parallel().map(x -> x.map(y -> Tensor.fromRGB(TestUtil.resize(y.get(), getImageSize()))));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  public Stream<LabeledObject<Tensor>> validationData() {
    return trainingData();
  }
  
  public int getImageSize() {
    return imageSize;
  }
  
  public List<String> getLabels() {
    if (null == labels) {
      synchronized (this) {
        if (null == labels) {
          labels = trainingData().map(x -> x.label).distinct().sorted().collect(Collectors.toList());
        }
      }
    }
    return labels;
  }
  
}
