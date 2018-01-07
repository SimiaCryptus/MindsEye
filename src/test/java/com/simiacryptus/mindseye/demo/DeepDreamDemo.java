/*
 * Copyright (c) 2018 by Andrew Charneski.
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

package com.simiacryptus.mindseye.demo;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.cudnn.CuDNN;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.models.VGG16;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.Step;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.TableOutput;
import com.simiacryptus.util.io.NotebookOutput;
import org.junit.Test;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * The type Image classifier run base.
 */
public class DeepDreamDemo extends NotebookReportBase {
  
  /**
   * Test.
   *
   * @throws Throwable the throwable
   */
  @Test
  public void run() {
    run(this::run);
  }
  
  /**
   * Test.
   *
   * @param log the log
   */
  public void run(NotebookOutput log) {
    
    
    String logName = "cuda_" + log.getName() + ".log";
    log.p(log.file((String) null, logName, "GPU Log"));
    CuDNN.apiLog = new PrintStream(log.file(logName));
    
    log.h1("Model");
    ImageClassifier vgg16 = log.code(() -> {
      return VGG16.fromS3_HDF5();
    });
    
    log.h1("Data");
    Tensor[] images = log.code(() -> {
      return Caltech101.trainingDataStream().sorted(getShuffleComparator()).map(labeledObj -> {
        BufferedImage img = labeledObj.data.get();
        img = TestUtil.resize(img, 224);
        return Tensor.fromRGB(img);
      }).limit(10).toArray(i1 -> new Tensor[i1]);
    });
    
    log.h1("Prediction");
    List<LinkedHashMap<String, Double>> predictions = log.code(() -> {
      return vgg16.predict(5, images);
    });
    
    List<String> vgg16Categories = vgg16.getCategories();
    for (int itemNumber = 0; itemNumber < images.length; itemNumber++) {
      log.h1("Image " + itemNumber);
      List<String> categories = predictions.get(itemNumber).keySet().stream().collect(Collectors.toList());
      log.p("Evolve from %s to %s", categories.get(0), categories.get(1));
      int targetCategoryIndex = vgg16Categories.indexOf(categories.get(1));
      SimpleLossNetwork supervised = new SimpleLossNetwork(vgg16.getNetwork().freeze(), new EntropyLossLayer());
      Trainable trainable = new ArrayTrainable(supervised, 1).setMask(true, false).setData(Arrays.<Tensor[]>asList(new Tensor[]{
        images[itemNumber], new Tensor(vgg16Categories.size()).set(targetCategoryIndex, 1.0)
      }));
      ArrayList<StepRecord> history = new ArrayList<>();
      log.code(() -> {
        new IterativeTrainer(trainable)
          .setMonitor(getTrainingMonitor(history))
          .setOrientation(new QQN())
          .setTimeout(15, TimeUnit.MINUTES)
          .run();
      });
      log.code(() -> {
        return TestUtil.plot(history);
      });
      try {
        log.p(log.image(images[itemNumber].toImage(), "result"));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    
    
    log.h1("Results");
    log.code(() -> {
      TableOutput tableOutput = new TableOutput();
      for (int i = 0; i < images.length; i++) {
        HashMap<String, Object> row = new HashMap<>();
        row.put("Image", log.image(images[i].toImage(), ""));
        row.put("Prediction", predictions.get(i).entrySet().stream()
                                         .map(e -> String.format("%s -> %.2f", e.getKey(), 100 * e.getValue()))
                                         .reduce((a, b) -> a + "<br/>" + b).get());
        tableOutput.putRow(row);
      }
      return tableOutput;
    }, 256 * 1024);
    log.setFrontMatterProperty("status", "OK");
  }
  
  public TrainingMonitor getTrainingMonitor(ArrayList<StepRecord> history) {
    TrainingMonitor monitor1 = TestUtil.getMonitor(history);
    return new TrainingMonitor() {
      @Override
      public void clear() {
        monitor1.clear();
      }
      
      @Override
      public void log(String msg) {
        TestUtil.originalOut.println(msg);
        monitor1.log(msg);
      }
      
      @Override
      public void onStepComplete(Step currentPoint) {
        monitor1.onStepComplete(currentPoint);
      }
    };
  }
  
  /**
   * Gets shuffle comparator.
   *
   * @param <T> the type parameter
   * @return the shuffle comparator
   */
  public <T> Comparator<T> getShuffleComparator() {
    final int seed = (int) ((System.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    return Comparator.comparingInt(a1 -> System.identityHashCode(a1) ^ seed);
  }
  
  /**
   * Gets target class.
   *
   * @return the target class
   */
  protected Class<?> getTargetClass() {
    return VGG16.class;
  }
  
  @Override
  public ReportType getReportType() {
    return ReportType.Demos;
  }
}
