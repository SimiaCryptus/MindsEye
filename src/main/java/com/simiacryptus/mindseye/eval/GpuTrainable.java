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

package com.simiacryptus.mindseye.eval;

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.java.PlaceholderLayer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.util.lang.TimedResult;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class handles dispatching network evaluations, and distributing the evaluations to the system GPU(s). This is
 * the main class the handles actual execution for training purposes.
 */
public class GpuTrainable implements DataTrainable, TrainableDataMask {
  
  /**
   * Between each iteration we have the option to initiate a garbage collection. This is a good opportunity since the
   * reachable object count will be at a minimum between collections, making GC more efficient. This can be configured
   * as a non-blocking operation by using the JVM flags "-XX:+ExplicitGCInvokesConcurrent -XX:+UseConcMarkSweepGC"
   */
  protected static boolean gcEachIteration = true;
  /**
   * The Gc period.
   */
  protected static double gcPeriod = 1.0;
  private static int verbosity = 0;
  /**
   * The Network.
   */
  protected final NNLayer network;
  /**
   * The Data.
   */
  protected List<Tensor[]> data;
  /**
   * The Mask.
   */
  boolean[] mask = null;
  private long lastGc = 0;
  
  /**
   * Instantiates a new Gpu trainable.
   *
   * @param network the network
   */
  public GpuTrainable(final NNLayer network) {
    this.network = network;
    data = null;
  }
  
  /**
   * Get nn context nn result [ ].
   *
   * @param data the data
   * @param mask the mask
   * @return the nn result [ ]
   */
  public static NNResult[] getNNContext(final List<Tensor[]> data, final boolean[] mask) {
    if (null == data) throw new IllegalArgumentException();
    if (0 >= data.size()) throw new IllegalArgumentException();
    final int cols = data.get(0).length;
    return IntStream.range(0, cols).parallel().mapToObj(col -> {
      final Tensor[] tensors = IntStream.range(0, data.size()).mapToObj(row -> data.get(row)[col]).toArray(i -> new Tensor[i]);
      if (null == mask || col >= mask.length || !mask[col]) {
        return new NNConstant(tensors);
      }
      else {
        return new NNResult(tensors) {
          @Override
          public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
            for (int index = 0; index < delta.length(); index++) {
              final Tensor dt = delta.get(index);
              final double[] d = dt.getData();
              final Tensor t = tensors[index];
              final double[] p = t.getData();
              buffer.get(new PlaceholderLayer<double[]>(p), p).addInPlace(d);
            }
          }
  
          @Override
          public boolean isAlive() {
            return true;
          }
        };
      }
    }).toArray(x1 -> new NNResult[x1]);
  }
  
  /**
   * Gets verbosity.
   *
   * @return the verbosity
   */
  public static int getVerbosity() {
    return verbosity;
  }
  
  /**
   * Sets verbose.
   *
   * @param verbose the verbose
   */
  public static void setVerbosity(final int verbose) {
    verbosity = verbose;
  }
  
  /**
   * Gets gc period.
   *
   * @return the gc period
   */
  public static double getGcPeriod() {
    return gcPeriod;
  }
  
  /**
   * Sets gc period.
   *
   * @param gcPeriod the gc period
   */
  public static void setGcPeriod(final double gcPeriod) {
    GpuTrainable.gcPeriod = gcPeriod;
  }
  
  /**
   * Is gc each iteration boolean.
   *
   * @return the boolean
   */
  public static boolean isGcEachIteration() {
    return gcEachIteration;
  }
  
  /**
   * Sets gc each iteration.
   *
   * @param gcEachIteration the gc each iteration
   */
  public static void setGcEachIteration(final boolean gcEachIteration) {
    GpuTrainable.gcEachIteration = gcEachIteration;
  }
  
  @Override
  public Tensor[][] getData() {
    return data.toArray(new Tensor[][]{});
  }
  
  @Override
  public boolean[] getMask() {
    return mask;
  }
  
  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    return measure(3, monitor);
  }
  
  /**
   * Eval point sample.
   *
   * @param list      the list
   * @param nncontext the nncontext
   * @param monitor   the monitor
   * @return the point sample
   */
  protected PointSample eval(final List<Tensor[]> list, final CudaExecutionContext nncontext, final TrainingMonitor monitor) {
    final TimedResult<PointSample> timedResult = TimedResult.time(() -> {
      final NNResult[] nnContext = GpuTrainable.getNNContext(list, mask);
      final NNResult result = network.eval(nncontext, nnContext);
      final TensorList resultData = result.getData();
      assert resultData.stream().allMatch(x -> x.dim() == 1);
      assert resultData.stream().allMatch(x -> Arrays.stream(x.getData()).allMatch(Double::isFinite));
      final DoubleSummaryStatistics statistics = resultData.stream().flatMapToDouble(x -> Arrays.stream(x.getData())).summaryStatistics();
      final double sum = statistics.getSum();
      final DeltaSet<NNLayer> xxx = new DeltaSet<NNLayer>();
      result.accumulate(xxx, 1.0);
      //log.info(String.format("Evaluated to %s delta buffers, %s mag", DeltaSet<NNLayer>.getMap().size(), DeltaSet<NNLayer>.getMagnitude()));
      return new PointSample(xxx, new StateSet<NNLayer>(xxx), sum, 0.0, list.size());
    });
    if (null != monitor && getVerbosity() > 0) {
      monitor.log(String.format("Device %s completed %s items in %.3f sec", nncontext.toString(), list.size(), timedResult.timeNanos / 1e9));
    }
    return timedResult.result.normalize();
  }
  
  @Override
  public Trainable setData(final List<Tensor[]> sampledData) {
    assert !sampledData.isEmpty();
    data = sampledData;
    return this;
  }
  
  /**
   * Sets data supplier.
   *
   * @param data the data
   */
  protected void setDataSupplier(final List<? extends Supplier<Tensor[]>> data) {
    setData(data.stream().parallel().map(x -> x.get()).collect(Collectors.toList()));
  }
  
  @Override
  public TrainableDataMask setMask(final boolean... mask) {
    this.mask = mask;
    return this;
  }
  
  /**
   * Measure point sample.
   *
   * @param retries the retries
   * @param monitor the monitor
   * @return the point sample
   */
  public PointSample measure(final int retries, final TrainingMonitor monitor) {
    try {
      assert !data.isEmpty();
  
      final TimedResult<PointSample> timedResult = TimedResult.time(() -> GpuController.INSTANCE.distribute(data,
                                                                                                            (list, dev) -> eval(list, dev, monitor),
                                                                                                            (a, b) -> a.addInPlace(b)
                                                                                                           ));
      //          log.info(String.format("Evaluated to %s delta arrays", DeltaSet<NNLayer>.run.size()));
      if (null != monitor && getVerbosity() > 1) {
        monitor.log(String.format("Evaluated %s items in %.4fs (%s/%s)", data.size(), timedResult.timeNanos / 1e9, timedResult.result.getMean(), timedResult.result.delta.getMagnitude()));
      }
      assert null != timedResult.result;
      // Between each iteration is a great time to collect garbage, since the reachable object count will be at a low point.
      // Recommended JVM flags: -XX:+ExplicitGCInvokesConcurrent -XX:+UseConcMarkSweepGC
      if (gcEachIteration && TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis() - lastGc) > gcPeriod) {
        lastGc = System.currentTimeMillis();
        GpuController.INSTANCE.cleanMemory();
      }
      return timedResult.result;
    } catch (final Exception e) {
      RecycleBin.DOUBLES.printNetProfiling(System.err);
      if (retries > 0) {
        lastGc = System.currentTimeMillis();
        GpuController.INSTANCE.cleanMemory();
        synchronized (CudaResource.gpuGeneration) {
          for (final Map.Entry<CudaExecutionContext, ExecutorService> entry : GpuController.INSTANCE.getGpuDriverThreads().asMap().entrySet()) {
            CudaResource.gpuGeneration.incrementAndGet();
            try {
              entry.getValue().submit(() -> {
                CudaPtr.getGpuStats(entry.getKey().getDeviceNumber()).usedMemory.set(0);
                return CuDNN.cudaDeviceReset();
              }).get();
            } catch (final InterruptedException e1) {
              throw new GpuError(e1);
            } catch (final ExecutionException e1) {
              throw new GpuError(e1);
            }
          }
        }
        CudaPtr.METRICS.invalidateAll();
        if (gcEachIteration && TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis() - lastGc) > gcPeriod) {
          lastGc = System.currentTimeMillis();
          GpuController.INSTANCE.cleanMemory();
        }
        return measure(retries - 1, monitor);
      }
      else {
        throw e;
      }
    }
  }
  
}
