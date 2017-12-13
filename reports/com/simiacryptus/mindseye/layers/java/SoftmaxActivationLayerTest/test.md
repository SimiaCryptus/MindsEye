# SoftmaxActivationLayer
## SoftmaxActivationLayerTest
### Json Serialization
Code from [StandardLayerTests.java:69](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L69) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    if ((echo == null)) throw new AssertionError("Failed to deserialize");
    if ((layer == echo)) throw new AssertionError("Serialization did not copy");
    if ((!layer.equals(echo))) throw new AssertionError("Serialization not equal");
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.SoftmaxActivationLayer",
      "id": "b8568925-014f-41e8-a0aa-0c9c3c5bf036",
      "isFrozen": false,
      "name": "SoftmaxActivationLayer/b8568925-014f-41e8-a0aa-0c9c3c5bf036"
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:153](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L153) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s\n--------------------\nDerivative: \n%s",
      Arrays.stream(inputPrototype).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get(),
      eval.getOutput().prettyPrint(),
      Arrays.stream(eval.getDerivative()).map(t -> t.prettyPrint()).reduce((a, b) -> a + ",\n" + b).get());
```

Returns: 

```
    --------------------
    Input: 
    [[ -0.236, 1.92, -0.384, -0.156 ]]
    --------------------
    Output: 
    [ 0.08633908043978972, 0.7456694008780793, 0.07446150940675879, 0.09353000927537213 ]
    --------------------
    Derivative: 
    [ 0.0, 0.0, 0.0, 0.0 ]
```



### Batch Execution
Code from [StandardLayerTests.java:102](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L102) executed in 0.00 seconds: 
```java
    return getBatchingTester().test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (80#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (40#)}
```



### Differential Validation
Code from [StandardLayerTests.java:110](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L110) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 1.76, -0.064, -0.412, 0.016 ]
    Inputs Statistics: {meanExponent=-0.7823225398782259, negative=2, min=0.016, max=0.016, mean=0.325, count=4.0, positive=2, stdDev=0.8439780802840794, zeros=0}
    Output: [ 0.689584714168348, 0.11128445187431771, 0.07857782640858947, 0.12055300754874468 ]
    Outputs Statistics: {meanExponent=-0.7846249554922285, negative=0, min=0.12055300754874468, max=0.12055300754874468, mean=0.24999999999999994, count=4.0, positive=4, stdDev=0.2542728896656876, zeros=0}
    Feedback for input 0
    Inputs Values: [ 1.76, -0.064, -0.412, 0.016 ]
    Value Statistics: {meanExponent=-0.7823225398782259, negative=2, min=0.016, max=0.016, mean=0.325, count=4.0, positive=2, stdDev=0.8439780802840794, zeros=0}
    Implemented Feedback: [ [ 0.21405763615370577, -0.07674005693713265, -0.05418606796393723, -0.08313151125263579 ], [ -0.07674005693713265, 0.09890022264535037, -0.008744490341355165, -0.013415675366862535 ], [ -0.05418606796393723, -0.008744490341355165, 0.07240335160549104, -0.009472793300198633
```
...[skipping 685 bytes](etc/104.txt)...
```
    00288037273 ] ]
    Measured Statistics: {meanExponent=-1.4137469270512164, negative=12, min=0.10602400288037273, max=0.10602400288037273, mean=2.2551405187698492E-13, count=16.0, positive=4, stdDev=0.08071934482729094, zeros=0}
    Feedback Error: [ [ -4.0583064994104134E-6, 1.4549105445404997E-6, 1.0273108330891012E-6, 1.5760858155799973E-6 ], [ -2.983055699956605E-6, 3.8444725272146485E-6, -3.399178722888335E-7, -5.214971508724076E-7 ], [ -2.283571984819377E-6, -3.6852040929437335E-7, 3.05130623780292E-6, -3.992134273607395E-7 ], [ -3.1544500085911586E-6, -5.090618203293373E-7, -3.5944815295410604E-7, 4.022960675753584E-6 ] ]
    Error Statistics: {meanExponent=-5.895923104084112, negative=10, min=4.022960675753584E-6, max=4.022960675753584E-6, mean=2.2550646246177752E-13, count=16.0, positive=6, stdDev=2.3390113820766395E-6, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.8721e-06 +- 1.4022e-06 [3.3992e-07 - 4.0583e-06] (16#)
    relativeTol: 1.7240e-05 +- 4.5476e-06 [9.4796e-06 - 2.1071e-05] (16#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=1.8721e-06 +- 1.4022e-06 [3.3992e-07 - 4.0583e-06] (16#), relativeTol=1.7240e-05 +- 4.5476e-06 [9.4796e-06 - 2.1071e-05] (16#)}
```



### Performance
Code from [StandardLayerTests.java:120](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L120) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    Evaluation performance: 0.000820s +- 0.000238s [0.000538s - 0.001252s]
    Learning performance: 0.000026s +- 0.000002s [0.000024s - 0.000029s]
    
```

