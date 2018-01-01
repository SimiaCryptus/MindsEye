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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.BasicTrainable;
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.orient.RecursiveSubspace;
import com.simiacryptus.mindseye.test.ProblemRun;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.integration.MnistProblemData;
import com.simiacryptus.mindseye.test.integration.OptimizationStrategy;
import com.simiacryptus.util.io.NotebookOutput;

import java.awt.*;
import java.util.List;
import java.util.function.Function;

/**
 * The type Compare qqn.
 */
public class Research extends OptimizerComparison {
  
  /**
   * The constant quadratic_quasi_newton.
   */
  public static OptimizationStrategy recursive_subspace = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Recursive Subspace method:");
    return log.code(() -> {
      final ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
             .setOrientation(new RecursiveSubspace() {
               @Override
               public void train(TrainingMonitor monitor, NNLayer subspace) {
                 //new SingleDerivativeTester(1e-3,1e-4).run(subspace, new Tensor[]{new Tensor()});
                 super.train(monitor, subspace);
               }
             })
             .setLineSearchFactory(name -> new StaticLearningRate(1.0));
      return trainer;
    });
  };
  /**
   * The constant recursive_subspace_2.
   */
  public static OptimizationStrategy recursive_subspace_2 = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Recursive Subspace method:");
    return log.code(() -> {
      final ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
             .setOrientation(new RecursiveSubspace() {
               @Override
               public void train(TrainingMonitor monitor, NNLayer subspace) {
                 //new SingleDerivativeTester(1e-3,1e-4).run(subspace, new Tensor[]{new Tensor()});
                 ArrayTrainable trainable = new ArrayTrainable(new BasicTrainable(subspace), new Tensor[][]{{new Tensor()}});
                 new IterativeTrainer(trainable)
                   .setOrientation(new QQN())
                   .setLineSearchFactory(n -> new QuadraticSearch())
                   .setMonitor(new TrainingMonitor() {
                     @Override
                     public void log(String msg) {
                       monitor.log("\t" + msg);
                     }
                   })
                   .setMaxIterations(getIterations()).setIterationsPerSample(getIterations()).run();
               }
             })
             .setLineSearchFactory(name -> new StaticLearningRate(1.0));
      return trainer;
    });
  };
  
  /**
   * The constant quadratic_quasi_newton.
   */
  public static OptimizationStrategy quadratic_quasi_newton = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Quadratic Quasi-Newton method:");
    return log.code(() -> {
      final ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
             .setOrientation(new com.simiacryptus.mindseye.opt.orient.QQN())
             .setLineSearchFactory(name -> new QuadraticSearch()
               .setCurrentRate(name.contains("QQN") ? 1.0 : 1e-6)
               .setRelativeTolerance(2e-1));
      return trainer;
    });
  };
  
  /**
   * The constant limited_memory_bfgs.
   */
  public static OptimizationStrategy limited_memory_bfgs = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Limited-Memory BFGS method:");
    return log.code(() -> {
      final ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMinTrainingSize(Integer.MAX_VALUE)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
             .setOrientation(new com.simiacryptus.mindseye.opt.orient.LBFGS())
             .setLineSearchFactory(name -> new QuadraticSearch()
               .setCurrentRate(name.contains("LBFGS") ? 1.0 : 1e-6));
      return trainer;
    });
  };
  
  /**
   * Instantiates a new Compare qqn.
   */
  public Research() {
    super(MnistTests.fwd_conv_2(), MnistTests.rev_conv_1, new MnistProblemData());
  }
  
  @Override
  public void compare(final NotebookOutput log, final Function<OptimizationStrategy, List<StepRecord>> test) {
    log.h1("Research Optimizer Comparison");
    
    log.h2("Recursive Subspace (Un-Normalized)");
    fwdFactory = MnistTests.fwd_conv_2();
    final ProblemRun subspace_1 = new ProblemRun("SS", test.apply(Research.recursive_subspace), Color.LIGHT_GRAY,
                                                 ProblemRun.PlotType.Line);
    
    log.h2("Recursive Subspace (Un-Normalized)");
    fwdFactory = MnistTests.fwd_conv_2();
    final ProblemRun subspace_2 = new ProblemRun("SS+QQN", test.apply(Research.recursive_subspace_2), Color.RED,
                                                 ProblemRun.PlotType.Line);
    
    log.h2("QQN (Normalized)");
    fwdFactory = MnistTests.fwd_conv_2(() -> new NormalizationMetaLayer());
    final ProblemRun qqn1 = new ProblemRun("QQN", test.apply(Research.quadratic_quasi_newton), Color.DARK_GRAY,
                                           ProblemRun.PlotType.Line);
    
    log.h2("L-BFGS (Strong Line Search) (Normalized)");
    fwdFactory = MnistTests.fwd_conv_2(() -> new NormalizationMetaLayer());
    final ProblemRun lbfgs_2 = new ProblemRun("LB-2", test.apply(Research.limited_memory_bfgs), Color.MAGENTA,
                                              ProblemRun.PlotType.Line);
    
    log.h2("L-BFGS (Normalized)");
    fwdFactory = MnistTests.fwd_conv_2(() -> new NormalizationMetaLayer());
    final ProblemRun lbfgs_1 = new ProblemRun("LB-1", test.apply(TextbookOptimizers.limited_memory_bfgs), Color.GREEN,
                                              ProblemRun.PlotType.Line);
    
    log.h2("L-BFGS-0 (Un-Normalized)");
    fwdFactory = MnistTests.fwd_conv_2();
    final ProblemRun rawlbfgs = new ProblemRun("LBFGS-0", test.apply(TextbookOptimizers.limited_memory_bfgs), Color.CYAN,
                                               ProblemRun.PlotType.Line);
    
    log.h2("Comparison");
    log.code(() -> {
      return TestUtil.compare("Convergence Plot", subspace_1, subspace_2, rawlbfgs, lbfgs_1, lbfgs_2, qqn1);
    });
    log.code(() -> {
      return TestUtil.compareTime("Convergence Plot", subspace_1, subspace_2, rawlbfgs, lbfgs_1, lbfgs_2, qqn1);
    });
  }
  
}
