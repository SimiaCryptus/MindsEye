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
import com.simiacryptus.mindseye.test.integration.ImageProblemData;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.LabeledObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Linear run.
 */
public class SupplementedProblemData implements ImageProblemData {
  
  private final ImageProblemData inner;
  private final Random random = new Random();
  private final int expansion = 10;
  
  /**
   * Instantiates a new Supplemented data.
   *
   * @param inner the inner
   */
  public SupplementedProblemData(ImageProblemData inner) {
    this.inner = inner;
  }
  
  /**
   * Print sample.
   *
   * @param log      the log
   * @param expanded the expanded
   * @param size     the size
   */
  public static void printSample(NotebookOutput log, Tensor[][] expanded, int size) {
    ArrayList<Tensor[]> list = new ArrayList<>(Arrays.asList(expanded));
    Collections.shuffle(list);
    log.p("Expanded Training Data Sample: " + list.stream().limit(size).map(x -> {
      try {
        return log.image(x[0].toGrayImage(), "");
      } catch (IOException e) {
        e.printStackTrace();
        return "";
      }
    }).reduce((a, b) -> a + b).get());
  }
  
  /**
   * Translate tensor.
   *
   * @param dx     the dx
   * @param dy     the dy
   * @param tensor the tensor
   * @return the tensor
   */
  protected static Tensor translate(int dx, int dy, Tensor tensor) {
    int sx = tensor.getDimensions()[0];
    int sy = tensor.getDimensions()[1];
    return new Tensor(tensor.coordStream().mapToDouble(c -> {
      int x = c.getCoords()[0] + dx;
      int y = c.getCoords()[1] + dy;
      if (x < 0 || x >= sx) {
        return 0.0;
      }
      else if (y < 0 || y >= sy) {
        return 0.0;
      }
      else {
        return tensor.get(x, y);
      }
    }).toArray(), tensor.getDimensions());
  }
  
  /**
   * Add noise tensor.
   *
   * @param tensor the tensor
   * @return the tensor
   */
  protected static Tensor addNoise(Tensor tensor) {
    return tensor.mapParallel((v) -> Math.random() < 0.9 ? v : (v + Math.random() * 100));
  }
  
  @Override
  public Stream<LabeledObject<Tensor>> validationData() throws IOException {
    return inner.validationData();
  }
  
  @Override
  public Stream<LabeledObject<Tensor>> trainingData() throws IOException {
    return inner.trainingData().flatMap(labeledObject -> {
      return IntStream.range(0, expansion).mapToObj(i -> {
        int dx = random.nextInt(10) - 5;
        int dy = random.nextInt(10) - 5;
        return addNoise(translate(dx, dy, labeledObject.data));
      }).map(t -> new LabeledObject<>(t, labeledObject.label));
    });
  }
}
