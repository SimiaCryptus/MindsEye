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

package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.indexing.INDArrayIndex;

public class INDArray {
  public INDArray dup(char c) {
    return null;
  }
  
  public int size(int i) {
    throw new RuntimeException();
  }
  
  public INDArray reshape(int size, int... i) {
    throw new RuntimeException("NI");
  }
  
  public INDArray dup() {
    throw new RuntimeException("NI");
  }
  
  public INDArray reshape(int miniBatchSize, int depth, int tsLength) {
    throw new RuntimeException("NI");
  }
  
  public int length() {
    throw new RuntimeException("NI");
  }
  
  public INDArray get(INDArrayIndex all, INDArrayIndex interval) {
    throw new RuntimeException("NI");
  }
  
  public int rows() {
    throw new RuntimeException("NI");
  }
  
  public int columns() {
    throw new RuntimeException("NI");
  }
  
  public void put(INDArrayIndex[] indArrayIndices, INDArray w_c) {
    throw new RuntimeException("NI");
  }
  
  public void putScalar(int i1, int i2, int i3, int i4, float v) {
    throw new RuntimeException("NI");
  }
  
  public void putScalar(int i1, int i2, int i3, float v) {
    throw new RuntimeException("NI");
  }
  
  public void putScalar(int i1, int i2, float v) {
    throw new RuntimeException("NI");
  }
  
  public void putScalar(int i1, float v) {
    throw new RuntimeException("NI");
  }
  
  public INDArray permute(int i, int i1, int i2, int i3) {
    throw new RuntimeException("NI");
  }
  
  public int tensorssAlongDimension(int i, int i1) {
    throw new RuntimeException("NI");
  }
  
  public INDArray tensorAlongDimension(int i, int i1, int i2) {
    throw new RuntimeException("NI");
  }
  
  public INDArray ravel() {
    throw new RuntimeException("NI");
  }
  
  public JsonNodeData data() {
    throw new RuntimeException("NI");
  }
  
  public int[] shape() {
    throw new RuntimeException("NI");
  }
  
  public INDArray muli(int i) {
    throw new RuntimeException("NI");
  }
  
  public void addi(INDArray newFilter) {
  }
  
  public INDArray permute(int i, int i1, int i2) {
    throw new RuntimeException("NI");
  }
  
  public INDArray reshape(int size, int size1, int size2, int i) {
    throw new RuntimeException("NI");
  }
  
  public INDArray reshape(int[] inputShape) {
    throw new RuntimeException("NI");
  }
  
  public INDArray getRow(int batch) {
    return null;
  }
  
  public float getFloat(int batch, int i) {
    throw new RuntimeException("NI");
  }
  
  public int getInt(int i, int i1) {
    throw new RuntimeException("NI");
  }
  
  public int rank() {
    throw new RuntimeException("NI");
  }
  
  public char ordering() {
    throw new RuntimeException("NI");
  }
}
