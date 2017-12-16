# ProductLayer
## ProductLayerTest
### Batch Execution
Code from [BatchingTester.java:66](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchingTester.java#L66) executed in 0.00 seconds: 
```java
    return test(reference, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



Code from [SingleDerivativeTester.java:77](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/SingleDerivativeTester.java#L77) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ -0.776, 0.0, -0.636 ]
    Inputs Statistics: {meanExponent=-0.15334058154669883, negative=2, min=-0.636, max=-0.636, mean=-0.4706666666666666, count=3.0, positive=0, stdDev=0.3376836126053828, zeros=1}
    Output: [ 0.0 ]
    Outputs Statistics: {meanExponent=NaN, negative=0, min=0.0, max=0.0, mean=0.0, count=1.0, positive=0, stdDev=0.0, zeros=1}
    
```

Returns: 

```
    java.lang.RuntimeException: com.simiacryptus.mindseye.lang.GpuError: java.util.concurrent.ExecutionException: com.simiacryptus.mindseye.lang.GpuError: Failed executing 1 items
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:133)
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.test(SingleDerivativeTester.java:77)
    	at com.simiacryptus.mindseye.test.unit.StandardLayerTests.lambda$test$2(StandardLayerTests.java:78)
    	at java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
    	at java.util.stream.ReferencePipeline$2$1.accept(ReferencePipeline.java:175)
    	at java.util.ArrayList$ArrayListSpliterator.forEachRemaining(Ar
```
...[skipping 6312 bytes](etc/319.txt)...
```
    ter$4.accumulate(SingleDerivativeTester.java:291)
    	at com.simiacryptus.mindseye.layers.java.ProductLayer$1.accumulate(ProductLayer.java:95)
    	at com.simiacryptus.mindseye.test.unit.SingleDerivativeTester.lambda$getFeedbackGradient$31(SingleDerivativeTester.java:313)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.lambda$evaluate$7(GpuController.java:184)
    	at java.util.stream.ReferencePipeline$3$1.accept(ReferencePipeline.java:193)
    	at java.util.Spliterators$ArraySpliterator.forEachRemaining(Spliterators.java:948)
    	at java.util.stream.AbstractPipeline.copyInto(AbstractPipeline.java:481)
    	at java.util.stream.AbstractPipeline.wrapAndCopyInto(AbstractPipeline.java:471)
    	at java.util.stream.ReduceOps$ReduceOp.evaluateSequential(ReduceOps.java:708)
    	at java.util.stream.AbstractPipeline.evaluate(AbstractPipeline.java:234)
    	at java.util.stream.ReferencePipeline.reduce(ReferencePipeline.java:479)
    	at com.simiacryptus.mindseye.layers.cudnn.GpuController.evaluate(GpuController.java:184)
    	... 5 more
    
```



