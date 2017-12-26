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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.MomentumStrategy;
import com.simiacryptus.mindseye.opt.orient.OwlQn;
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
 * The type Textbook optimizers.
 */
public class TextbookOptimizers extends OptimizerComparison {
  
  /**
   * The constant conjugate_gradient_descent.
   */
  public static OptimizationStrategy conjugate_gradient_descent = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Conjugate Gradient Descent method:");
    return log.code(() -> {
      final ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMinTrainingSize(Integer.MAX_VALUE)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
        .setOrientation(new GradientDescent())
        .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-5));
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
        .setLineSearchFactory(name -> new ArmijoWolfeSearch()
          .setAlpha(name.contains("LBFGS") ? 1.0 : 1e-6));
      return trainer;
    });
  };
  /**
   * The constant orthantwise_quasi_newton.
   */
  public static OptimizationStrategy orthantwise_quasi_newton = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Orthantwise Quasi-Newton search method:");
    return log.code(() -> {
      final ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMinTrainingSize(Integer.MAX_VALUE)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
        .setOrientation(new OwlQn())
        .setLineSearchFactory(name -> new ArmijoWolfeSearch()
          .setAlpha(name.contains("OWL") ? 1.0 : 1e-6));
      return trainer;
    });
  };
  /**
   * The constant simple_gradient_descent.
   */
  public static OptimizationStrategy simple_gradient_descent = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Stochastic Gradient Descent method:");
    return log.code(() -> {
      final double rate = 0.05;
      final ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMinTrainingSize(Integer.MAX_VALUE)
        .setMaxEpochIterations(100)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
        .setOrientation(new GradientDescent())
        .setLineSearchFactory(name -> new StaticLearningRate(rate));
      return trainer;
    });
  };
  /**
   * The constant stochastic_gradient_descent.
   */
  public static OptimizationStrategy stochastic_gradient_descent = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Stochastic Gradient Descent method with momentum and adaptve learning rate:");
    return log.code(() -> {
      final double carryOver = 0.5;
      final ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMaxEpochIterations(100)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
        .setOrientation(new MomentumStrategy(new GradientDescent()).setCarryOver(carryOver))
        .setLineSearchFactory(name -> new ArmijoWolfeSearch());
      return trainer;
    });
  };
  
  /**
   * Instantiates a new Compare textbook.
   */
  public TextbookOptimizers() {
    super(MnistTests.fwd_linear_1, MnistTests.rev_linear_1, new MnistProblemData());
  }
  
  @Override
  public void compare(final NotebookOutput log, final Function<OptimizationStrategy, List<StepRecord>> test) {
    log.h1("Textbook Optimizer Comparison");
    log.h2("GD");
    final ProblemRun gd = new ProblemRun("GD", Color.BLACK,
      test.apply(TextbookOptimizers.simple_gradient_descent), ProblemRun.PlotType.Line);
    log.h2("SGD");
    final ProblemRun sgd = new ProblemRun("SGD", Color.GREEN,
      test.apply(TextbookOptimizers.stochastic_gradient_descent), ProblemRun.PlotType.Line);
    log.h2("CGD");
    final ProblemRun cgd = new ProblemRun("CjGD", Color.BLUE,
      test.apply(TextbookOptimizers.conjugate_gradient_descent), ProblemRun.PlotType.Line);
    log.h2("L-BFGS");
    final ProblemRun lbfgs = new ProblemRun("L-BFGS", Color.MAGENTA,
      test.apply(TextbookOptimizers.limited_memory_bfgs), ProblemRun.PlotType.Line);
    log.h2("OWL-QN");
    final ProblemRun owlqn = new ProblemRun("OWL-QN", Color.ORANGE,
      test.apply(TextbookOptimizers.orthantwise_quasi_newton), ProblemRun.PlotType.Line);
    log.h2("Comparison");
    log.code(() -> {
      return TestUtil.compare("Convergence Plot", gd, sgd, cgd, lbfgs, owlqn);
    });
    log.code(() -> {
      return TestUtil.compareTime("Convergence Plot", gd, sgd, cgd, lbfgs, owlqn);
    });
  }
}
