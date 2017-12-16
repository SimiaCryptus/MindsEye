# StdDevMetaLayer
## StdDevMetaLayerTest
### Network Diagram
This is a network with the following layout:

Code from [StandardLayerTests.java:72](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/StandardLayerTests.java#L72) executed in 0.25 seconds: 
```java
    return Graphviz.fromGraph(TestUtil.toGraph((DAGNetwork) layer))
      .height(400).width(600).render(Format.PNG).toImage();
```

Returns: 

![Result](etc/test.244.png)



### Differential Validation
Code from [BatchDerivativeTester.java:76](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/unit/BatchDerivativeTester.java#L76) executed in 0.00 seconds: 
```java
    return test(component, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ],
    [ 1.78, 1.364, 0.512 ]
    Inputs Statistics: {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0},
    {meanExponent=0.03150144453506163, negative=0, min=0.512, max=0.512, mean=1.2186666666666668, count=3.0, positive=3, stdDev=0.5277608886186579, zeros=0}
    Output: [ 0.5277608886186579 ]
    Outputs Statistics: {meanExponent=-0.2775627976826158, negative=0, min=0.5277608886186579, max=0.5277608886186579, mean=0.5277608886186579, count=1.0, positive=1, stdDev=0.0, zeros=0}
    
```

Returns: 

```
    java.lang.RuntimeException: com.simiacryptus.mindseye.lang.GpuError: java.util.concurrent.ExecutionException: com.simiacryptus.mindseye.lang.GpuError: Failed executing 1 items
    	at com.simiacryptus.util.lang.TimedResult.time(TimedResult.java:61)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.lambda$code$2(MarkdownNotebookOutput.java:138)
    	at com.simiacryptus.util.test.SysOutInterceptor.withOutput(SysOutInterceptor.java:72)
    	at com.simiacryptus.util.io.MarkdownNotebookOutput.code(MarkdownNotebookOutput.java:136)
    	at com.simiacryptus.util.io.NotebookOutput.code(NotebookOutput.java:133)
    	at com.simiacryptus.mindseye.test.unit.BatchDerivativeTester.test(BatchDerivativeTester.java:76)
    	at com.simiacryptus.mindseye.test.unit.StandardLayerTests.lambda$test$2(StandardLayerTests.java:78)
    	at java.util.stream.ForEachOps$ForEachOp$OfRef.accept(ForEachOps.java:184)
    	at java.util.stream.ReferencePipeline$2$1.accept(ReferencePipeline.java:175)
    	at java.util.ArrayList$ArrayListSpliterator.forEachRemaining(Arra
```
...[skipping 7581 bytes](etc/341.txt)...
```
    r$1.accumulate(NthPowerActivationLayer.java:172)
    	at com.simiacryptus.mindseye.network.CountingNNResult.accumulate(CountingNNResult.java:108)
    	at com.simiacryptus.mindseye.test.unit.BatchDerivativeTester.lambda$getFeedbackGradient$33(BatchDerivativeTester.java:299)
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



