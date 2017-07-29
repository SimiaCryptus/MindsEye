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

package com.simiacryptus.mindseye;

import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.activation.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.synapse.BiasLayer;
import com.simiacryptus.mindseye.layers.synapse.DenseSynapseLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.region.SingleOrthant;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.trainable.L12Normalizer;
import com.simiacryptus.mindseye.opt.trainable.ScheduledSampleTrainable;
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable;
import com.simiacryptus.mindseye.opt.trainable.Trainable;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.ml.Tensor;
import com.simiacryptus.util.test.MNIST;
import com.simiacryptus.util.test.TestCategories;
import com.simiacryptus.util.text.TableOutput;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class MNistDemo {

  public static class Logic {
  
    public void validate(NotebookOutput log, PipelineNetwork network) {
      log.p("If we test our model against the entire validation dataset, we get this accuracy:");
      log.code(()->{
        try {
          return MNIST.validationDataStream().mapToDouble(labeledObject->{
            int actualCategory = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
            double[] predictionSignal = network.eval(labeledObject.data).data.get(0).getData();
            int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
            return predictionList[0]==actualCategory?1:0;
          }).average().getAsDouble() * 100;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
  
      log.p("Let's examine some incorrectly predicted results in more detail:");
      log.code(()->{
        try {
          TableOutput table = new TableOutput();
          MNIST.validationDataStream().map(labeledObject->{
            try {
              int actualCategory = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
              double[] predictionSignal = network.eval(labeledObject.data).data.get(0).getData();
              int[] predictionList = IntStream.range(0, 10).mapToObj(x -> x).sorted(Comparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x).toArray();
              if(predictionList[0] == actualCategory) return null; // We will only examine mispredicted rows
              LinkedHashMap<String, Object> row = new LinkedHashMap<String, Object>();
              row.put("Image", log.image(labeledObject.data.toGrayImage(),labeledObject.label));
              row.put("Prediction", Arrays.stream(predictionList).limit(3)
                                        .mapToObj(i->String.format("%d (%.1f%%)",i, 100.0*predictionSignal[i]))
                                        .reduce((a,b)->a+", "+b).get());
              return row;
            } catch (IOException e) {
              throw new RuntimeException(e);
            }
          }).filter(x->null!=x).limit(100).forEach(table::putRow);
          return table;
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    }
  
    public void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData) {
      log.p("Training a model involves a few different components. First, our model is combined with a loss function. " +
                "Then we take that model and combine it with our training data to define a trainable object. " +
                "Finally, we use a simple iterative scheme to refine the weights of our model. " +
                "The final output is the last output value of the loss function when evaluating the last batch.");
      log.code(()->{
        SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
        StochasticArrayTrainable trainable = new StochasticArrayTrainable(trainingData, supervisedNetwork, 1000);
        return new IterativeTrainer(trainable)
            .setTimeout(5, TimeUnit.MINUTES)
            .setMaxIterations(500)
            .run();
      });
    }
  
    public Tensor[][] getTrainingData(NotebookOutput log) {
      log.p("We use the standard MNIST dataset, made available by a helper function. " +
                "In order to use data, we convert it into data tensors; helper functions are defined to " +
                "work with images.");
      return log.code(() -> {
        try {
          return MNIST.trainingDataStream().map(labeledObject -> {
            Tensor categoryTensor = new Tensor(10);
            int category = Integer.parseInt(labeledObject.label.replaceAll("[^\\d]", ""));
            categoryTensor.set(category, 1);
            return new Tensor[]{labeledObject.data, categoryTensor};
          }).toArray(i->new Tensor[i][]);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    }
  
    public PipelineNetwork buildModel(NotebookOutput log) {
      log.p("This is a very simple model that performs basic logistic regression. " +
                "It is expected to be trainable to about 91% accuracy on MNIST.");
      return log.code(()->{
        PipelineNetwork network = new PipelineNetwork();
        network.add(new BiasLayer(28,28,1));
        network.add(new DenseSynapseLayer(new int[]{28,28,1},new int[]{10})
          .setWeights(()->0.001*(Math.random()-0.45)));
        network.add(new SoftmaxActivationLayer());
        return network;
      });
    }
  }
  
  @Test
  @Category(TestCategories.Report.class)
  public void basic() throws IOException {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this).addCopy(System.out)) {
      log.p("First, define a model:");
      Logic testLogic = new Logic();
      PipelineNetwork network = testLogic.buildModel(log);
      Tensor[][] trainingData = testLogic.getTrainingData(log);
      testLogic.train(log, network, trainingData);
      testLogic.validate(log, network);
    }
  }
  
  @Test
  @Category(TestCategories.Report.class)
  public void bellsAndWhistles() throws IOException {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this).addCopy(System.out)) {
      log.p("First, define a model:");
      Logic testLogic = new Logic(){
        @Override
        public void train(NotebookOutput log, PipelineNetwork network, Tensor[][] trainingData) {
          log.code(()->{
            SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
            Trainable trainable = ScheduledSampleTrainable.Pow(trainingData, supervisedNetwork, 1000, 1, 0.0);
            L12Normalizer normalizer = new L12Normalizer(trainable) {
              @Override
              protected double getL1(NNLayer layer) {
                if(layer instanceof DenseSynapseLayer) return 0.001;
                return 0;
              }
  
              @Override
              protected double getL2(NNLayer layer) {
                return 0;
              }
            };
            IterativeTrainer trainer = new IterativeTrainer(normalizer);
            trainer.setLineSearchFactory((s)->new ArmijoWolfeSearch().setC1(1e-4).setC2(0.9));
            trainer.setOrientation(new TrustRegionStrategy(new LBFGS().setMinHistory(5)) {
              @Override
              public TrustRegion getRegionPolicy(NNLayer layer) {
                if(layer instanceof DenseSynapseLayer) return new SingleOrthant();
                return null;
              }
            });
            trainer.setMonitor(new TrainingMonitor(){
              @Override
              public void log(String msg) {
                System.out.print(msg);
              }
  
              @Override
              public void onStepComplete(Step currentPoint) {
                super.onStepComplete(currentPoint);
              }
            });
            trainer.setTimeout(5, TimeUnit.MINUTES).run();
          });
        }
      };
      PipelineNetwork network = testLogic.buildModel(log);
      Tensor[][] trainingData = testLogic.getTrainingData(log);
      testLogic.train(log, network, trainingData);
      testLogic.validate(log, network);
    }
  }
}
