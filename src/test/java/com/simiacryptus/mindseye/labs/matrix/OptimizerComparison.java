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

import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.layers.java.NormalizationMetaLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.mindseye.opt.orient.RecursiveSubspace;
import com.simiacryptus.mindseye.test.ProblemRun;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.awt.*;
import java.io.IOException;
import java.util.List;
import java.util.function.Function;

/**
 * The type Optimizer comparison.
 */
public abstract class OptimizerComparison {
  
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
            //new SingleDerivativeTester(1e-3,1e-4).test(subspace, new Tensor[]{new Tensor()});
            super.train(monitor, subspace);
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
   * The Data.
   */
  protected ImageProblemData data;
  /**
   * The Fwd factory.
   */
  protected FwdNetworkFactory fwdFactory;
  /**
   * The Rev factory.
   */
  protected RevNetworkFactory revFactory;
  private int timeoutMinutes = 15;
  
  /**
   * Instantiates a new Optimizer comparison.
   *
   * @param fwdFactory the fwd factory
   * @param revFactory the rev factory
   * @param data       the data
   */
  public OptimizerComparison(final FwdNetworkFactory fwdFactory, final RevNetworkFactory revFactory, final ImageProblemData data) {
    this.fwdFactory = fwdFactory;
    this.revFactory = revFactory;
    this.data = data;
  }
  
  /**
   * Classification comparison.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void classification() throws IOException {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != TestUtil.originalOut) {
        log.addCopy(TestUtil.originalOut);
      }
      compare(log, opt -> {
        return new ClassifyProblem(fwdFactory, opt, data, 10)
          .setTimeoutMinutes(timeoutMinutes).run(log).getHistory();
      });
    }
  }
  
  /**
   * Compare.
   *
   * @param log  the log
   * @param test the test
   */
  public abstract void compare(NotebookOutput log, Function<OptimizationStrategy, List<StepRecord>> test);
  
  /**
   * Encoding comparison.
   *
   * @throws IOException the io exception
   */
  @Test
  @Ignore
  @Category(TestCategories.Report.class)
  public void encoding() throws IOException {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != TestUtil.originalOut) {
        log.addCopy(TestUtil.originalOut);
      }
      compare(log, opt -> {
        return new EncodingProblem(revFactory, opt, data, 10)
          .setTimeoutMinutes(timeoutMinutes).setTrainingSize(5000).run(log).getHistory();
      });
    }
  }
  
  /**
   * Gets timeout minutes.
   *
   * @return the timeout minutes
   */
  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }
  
  /**
   * Sets timeout minutes.
   *
   * @param timeoutMinutes the timeout minutes
   * @return the timeout minutes
   */
  public OptimizerComparison setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
  
  /**
   * The type Compare qqn.
   */
  public static class Research extends OptimizerComparison {
  
    /**
     * Instantiates a new Compare qqn.
     */
    public Research() {
      super(MnistTests.fwd_conv_2(), MnistTests.rev_conv_1, new MnistProblemData());
    }

    @Override
    public void compare(final NotebookOutput log, final Function<OptimizationStrategy, List<StepRecord>> test) {
      log.h1("Research Optimizer Comparison");
  
      log.h2("R-L-FBGS (Un-Normalized)");
      fwdFactory = MnistTests.fwd_conv_2();
      final ProblemRun rlbfgs = new ProblemRun("R-L-FBGS", Color.RED,
        test.apply(OptimizerComparison.recursive_subspace), ProblemRun.PlotType.Line);
  
      log.h2("QQN (Normalized)");
      fwdFactory = MnistTests.fwd_conv_2(() -> new NormalizationMetaLayer());
      final ProblemRun qqn1 = new ProblemRun("QQN", Color.DARK_GRAY,
        test.apply(OptimizerComparison.quadratic_quasi_newton), ProblemRun.PlotType.Line);
  
      log.h2("L-BFGS (Normalized)");
      fwdFactory = MnistTests.fwd_conv_2(() -> new NormalizationMetaLayer());
      final ProblemRun lbfgs_1 = new ProblemRun("LBFGS", Color.GREEN,
        test.apply(TextbookOptimizers.limited_memory_bfgs), ProblemRun.PlotType.Line);
  
      log.h2("L-BFGS-2 (Normalized)");
      fwdFactory = MnistTests.fwd_conv_2(() -> new NormalizationMetaLayer());
      final ProblemRun lbfgs_2 = new ProblemRun("LBFGS-2", Color.MAGENTA,
        test.apply(OptimizerComparison.limited_memory_bfgs), ProblemRun.PlotType.Line);
  
      log.h2("L-BFGS (Un-Normalized)");
      fwdFactory = MnistTests.fwd_conv_2();
      final ProblemRun rawlbfgs = new ProblemRun("Raw LBFGS", Color.CYAN,
        test.apply(TextbookOptimizers.limited_memory_bfgs), ProblemRun.PlotType.Line);
      
      log.h2("Comparison");
      log.code(() -> {
        return TestUtil.compare("Convergence Plot", rlbfgs, rawlbfgs, lbfgs_1, qqn1);
      });
      log.code(() -> {
        return TestUtil.compareTime("Convergence Plot", rlbfgs, rawlbfgs, lbfgs_1, qqn1);
      });
    }

  }
  
  /**
   * The type Compare textbook.
   */
  public static class Textbook extends OptimizerComparison {
  
    /**
     * Instantiates a new Compare textbook.
     */
    public Textbook() {
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
}
