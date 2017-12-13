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
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.test.*;
import com.simiacryptus.util.io.MarkdownNotebookOutput;
import com.simiacryptus.util.io.NotebookOutput;
import com.simiacryptus.util.test.TestCategories;
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
  public static OptimizationStrategy quadratic_quasi_newton = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Quadratic Quasi-Newton method:");
    return log.code(() -> {
      ValidatingTrainer trainer = new ValidatingTrainer(trainingSubject, validationSubject)
        .setMonitor(monitor);
      trainer.getRegimen().get(0)
        .setOrientation(new com.simiacryptus.mindseye.opt.orient.QQN())
        .setLineSearchFactory(name -> new QuadraticSearch()
          .setCurrentRate(name.contains("QQN") ? 1.0 : 1e-6)
          .setRelativeTolerance(2e-1));
      return trainer;
    });
  };
  private final FwdNetworkFactory fwdFactory;
  private final RevNetworkFactory revFactory;
  private final ImageProblemData data;
  private int timeoutMinutes = 1;
  /**
   * Instantiates a new Optimizer comparison.
   *
   * @param fwdFactory the fwd factory
   * @param revFactory the rev factory
   * @param data       the data
   */
  public OptimizerComparison(FwdNetworkFactory fwdFactory, RevNetworkFactory revFactory, ImageProblemData data) {
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
      if (null != TestUtil.originalOut) log.addCopy(TestUtil.originalOut);
      compare(log, opt -> {
        return new ClassifyProblem(fwdFactory, opt, data, 10)
          .setTimeoutMinutes(timeoutMinutes).run(log).getHistory();
      });
    }
  }
  
  /**
   * Encoding comparison.
   *
   * @throws IOException the io exception
   */
  @Test
  @Category(TestCategories.Report.class)
  public void encoding() throws IOException {
    try (NotebookOutput log = MarkdownNotebookOutput.get(this)) {
      if (null != TestUtil.originalOut) log.addCopy(TestUtil.originalOut);
      compare(log, opt -> {
        return new EncodingProblem(revFactory, opt, data, 10)
          .setTimeoutMinutes(timeoutMinutes).setTrainingSize(5000).run(log).getHistory();
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
  public OptimizerComparison setTimeoutMinutes(int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }
  
  /**
   * The type Compare qqn.
   */
  public static class CompareQQN extends OptimizerComparison {
    
    /**
     * Instantiates a new Compare qqn.
     */
    public CompareQQN() {
      super(MnistTests.fwd_conv_1, MnistTests.rev_conv_1, new MnistProblemData());
    }
    
    @Override
    public void compare(NotebookOutput log, Function<OptimizationStrategy, List<StepRecord>> test) {
      log.h1("QQN-LBFGS Comparison");
      log.h2("L-BFGS");
      ProblemRun lbfgs = new ProblemRun("LBFGS", Color.BLUE, test.apply(TextbookOptimizers.limited_memory_bfgs));
      log.h2("QQN");
      ProblemRun qqn = new ProblemRun("QQN", Color.GREEN, test.apply(quadratic_quasi_newton));
      log.h2("Comparison");
      log.code(() -> {
        return TestUtil.compare(lbfgs, qqn);
      });
      log.code(() -> {
        return TestUtil.compareTime(lbfgs, qqn);
      });
    }
    
  }
  
  /**
   * The type Compare textbook.
   */
  public static class CompareTextbook extends OptimizerComparison {
    
    /**
     * Instantiates a new Compare textbook.
     */
    public CompareTextbook() {
      super(MnistTests.fwd_linear_1, MnistTests.rev_linear_1, new MnistProblemData());
    }
    
    @Override
    public void compare(NotebookOutput log, Function<OptimizationStrategy, List<StepRecord>> test) {
      log.h1("Textbook Optimizer Comparison");
      log.h2("GD");
      ProblemRun gd = new ProblemRun("GD", Color.BLACK, test.apply(TextbookOptimizers.simple_gradient_descent));
      log.h2("SGD");
      ProblemRun sgd = new ProblemRun("SGD", Color.GREEN, test.apply(TextbookOptimizers.stochastic_gradient_descent));
      log.h2("CGD");
      ProblemRun cgd = new ProblemRun("CjGD", Color.BLUE, test.apply(TextbookOptimizers.conjugate_gradient_descent));
      log.h2("L-BFGS");
      ProblemRun lbfgs = new ProblemRun("L-BFGS", Color.MAGENTA, test.apply(TextbookOptimizers.limited_memory_bfgs));
      log.h2("OWL-QN");
      ProblemRun owlqn = new ProblemRun("OWL-QN", Color.ORANGE, test.apply(TextbookOptimizers.orthantwise_quasi_newton));
      log.h2("Comparison");
      log.code(() -> {
        return TestUtil.compare(gd, sgd, cgd, lbfgs, owlqn);
      });
      log.code(() -> {
        return TestUtil.compareTime(gd, sgd, cgd, lbfgs, owlqn);
      });
    }
    
  }
}
