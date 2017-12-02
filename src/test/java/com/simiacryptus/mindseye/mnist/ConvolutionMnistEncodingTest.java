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

package com.simiacryptus.mindseye.mnist;

import com.simiacryptus.mindseye.data.MNIST;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.f64.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.util.io.NotebookOutput;

public class ConvolutionMnistEncodingTest extends MnistEncodingTest {
  @Override
  public DAGNetwork buildModel(NotebookOutput log) {
    return log.code(() -> {
      PipelineNetwork network = new PipelineNetwork(2);
      DAGNode input = network.getInput(0);
      network.add(new ConvolutionLayer(3,3,8).setWeights(()->0.1*(Math.random()-0.5)));
      network.add(new ImgReshapeLayer(2,2,true));
      network.add(new ReLuActivationLayer());
      network.add(new ConvolutionLayer(3,3,2).setWeights(()->0.1*(Math.random()-0.5)));
      network.add(new ReLuActivationLayer());
//      network.add(new FullyConnectedLayer(new int[]{features}, new int[]{28, 28, 4})
//        .setWeights(() -> 0.25 * (Math.random() - 0.5)), input);
//      network.add(new LinearActivationLayer());
      DAGNode image = network.add("image", new BiasLayer(28, 28, 1), network.getHead());
      DAGNode softmax = network.add(new SoftmaxActivationLayer(), input);
      
      network.add(new SumInputsLayer(),
        network.add(new LinearActivationLayer().setScale(1).freeze(),
          network.add(new EntropyLossLayer(), softmax, softmax)),
        network.add(new NthPowerActivationLayer().setPower(1.0 / 2.0),
          network.add(new MeanSqLossLayer(), image, network.getInput(1))
        )
      )
      ;
      
      return network;
    });
  }
  
  @Override
  public Tensor[][] getTrainingData(NotebookOutput log) {
    return log.code(() -> {
      return MNIST.trainingDataStream().map(labeledObject -> {
        return new Tensor[]{new Tensor(14,14,1).fill(() -> 0.5 * (Math.random() - 0.5)), labeledObject.data};
      }).toArray(i -> new Tensor[i][]);
    });
  }
}
