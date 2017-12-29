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

package org.deeplearning4j.nn.conf.inputs;

public class InputType {
  ;
  
  public static InputType feedForward(int i) {
    return null;
  }
  
  public static InputType recurrent(int i) {
    throw new RuntimeException("NI");
  }
  
  public static InputType convolutional(int i, int i1, int i2) {
    throw new RuntimeException("NI");
  }
  
  public static class InputTypeConvolutional extends InputType {
    private int height;
    private int width;
    private int depth;
    
    public InputTypeConvolutional(int i, int i1, int i2) {
      super();
    }
    
    public int getHeight() {
      return height;
    }
    
    public int getWidth() {
      return width;
    }
    
    public int getDepth() {
      return depth;
    }
  }
  
  public static class InputTypeRecurrent extends InputType {
    private int size;
    private int timeSeriesLength;
    
    public InputTypeRecurrent(int i) {
      super();
    }
    
    public int getSize() {
      return size;
    }
    
    public int getTimeSeriesLength() {
      return timeSeriesLength;
    }
  }
  
  public static class InputTypeFeedForward extends InputType {
    private int size;
    
    public InputTypeFeedForward(int i) {
    
    }
    
    public int getSize() {
      return size;
    }
  }
}
