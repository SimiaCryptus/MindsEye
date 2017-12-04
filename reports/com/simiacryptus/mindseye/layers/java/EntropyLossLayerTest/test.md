# EntropyLossLayer
## EntropyLossLayerTest
### Json Serialization
Code from [LayerTestBase.java:83](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L83) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.EntropyLossLayer",
      "id": "a864e734-2f23-44db-97c1-504000002bad",
      "isFrozen": false,
      "name": "EntropyLossLayer/a864e734-2f23-44db-97c1-504000002bad"
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:120](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L120) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ 0.7772205324205131, 0.1476715216901574, 0.9023115935011684, 0.6288745384253454 ],
    [ 0.15006825613884756, 0.4346230571004488, 0.46757215503490357, 0.403969574261284 ]]
    --------------------
    Output: 
    [ 1.1045884479274184 ]
```



### Batch Execution
Code from [LayerTestBase.java:138](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L138) executed in 0.00 seconds: 
```java
    BatchingTester batchingTester = getBatchingTester();
    return batchingTester==null?null:batchingTester.test(layer, inputPrototype);
```

Returns: 

```
    ToleranceStatistics{absoluteTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (90#), relativeTol=0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (54#)}
```



### Differential Validation
Code from [LayerTestBase.java:144](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.00 seconds: 
```java
    return getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Inputs: [ 0.7772205324205131, 0.1476715216901574, 0.9023115935011684, 0.6288745384253454 ],
    [ 0.15006825613884756, 0.4346230571004488, 0.46757215503490357, 0.403969574261284 ]
    Inputs Statistics: {meanExponent=-0.2965596089088781, negative=0, min=0.6288745384253454, max=0.6288745384253454, mean=0.6140195465092961, count=4.0, positive=4, stdDev=0.2861153688631103, zeros=0},
    {meanExponent=-0.4773502762964599, negative=0, min=0.403969574261284, max=0.403969574261284, mean=0.3640582606338709, count=4.0, positive=4, stdDev=0.12557781632685341, zeros=0}
    Output: [ 1.1045884479274184 ]
    Outputs Statistics: {meanExponent=0.04320049694703621, negative=0, min=1.1045884479274184, max=1.1045884479274184, mean=1.1045884479274184, count=1.0, positive=1, stdDev=0.0, zeros=0}
    Feedback for input 0
    Inputs Values: [ 0.7772205324205131, 0.1476715216901574, 0.9023115935011684, 0.6288745384253454 ]
    Value Statistics: {meanExponent=-0.2965596089088781, negative=0, min=0.6288745384253454, max=0.6288745384253454, mean=0.6140195465092961, count=4.0, positive=4, stdDev=0.2861153688631103, zeros=0}
    Implemented Feedback: [ [ -0.1930832368407549 ], [ -2.9431745005808883 ], [ -0.5181936687975163 ], [ -0.6423691047705533 ] ]
    Implemented Statistics: {meanExponent=-0.18079066738758176, negative=4, min=-0.6423691047705533, max=-0.6423691047705533, mean=-1.0742051277474283, count=4.0, positive=0, stdDev=1.0914501129745398, zeros=0}
    Measured Feedback: [ [ -0.193083216082357 ], [ -2.9431743930885546 ], [ -0.5181936657550068 ], [ -0.6423690912527036 ] ]
    Measured Statistics: {meanExponent=-0.18079068594800393, negative=4, min=-0.6423690912527036, max=-0.6423690912527036, mean=-1.0742050915446555, count=4.0, positive=0, stdDev=1.0914500728719199, zeros=0}
    Feedback Error: [ [ 2.075839788950873E-8 ], [ 1.0749233370077604E-7 ], [ 3.042509466588683E-9 ], [ 1.3517849728472697E-8 ] ]
    Error Statistics: {meanExponent=-7.759322280934526, negative=0, min=1.3517849728472697E-8, max=1.3517849728472697E-8, mean=3.620277269633654E-8, count=4.0, positive=4, stdDev=4.1638140595202995E-8, zeros=0}
    Feedback for input 1
    Inputs Values: [ 0.15006825613884756, 0.4346230571004488, 0.46757215503490357, 0.403969574261284 ]
    Value Statistics: {meanExponent=-0.4773502762964599, negative=0, min=0.403969574261284, max=0.403969574261284, mean=0.3640582606338709, count=4.0, positive=4, stdDev=0.12557781632685341, zeros=0}
    Implemented Feedback: [ [ 0.2520311433763037 ], [ 1.9127649198826724 ], [ 0.10279537123216684 ], [ 0.4638235041397644 ] ]
    Implemented Statistics: {meanExponent=-0.40963947005268775, negative=0, min=0.4638235041397644, max=0.4638235041397644, mean=0.6828537346577268, count=4.0, positive=4, stdDev=0.7215836014559508, zeros=0}
    Measured Feedback: [ [ 0.252031151504184 ], [ 1.9127649242278721 ], [ 0.1027953722143593 ], [ 0.46382351293061674 ] ]
    Measured Statistics: {meanExponent=-0.40963946320939715, negative=0, min=0.46382351293061674, max=0.46382351293061674, mean=0.682853740219258, count=4.0, positive=4, stdDev=0.7215836012298315, zeros=0}
    Feedback Error: [ [ 8.127880257724485E-9 ], [ 4.345199755562135E-9 ], [ 9.821924662478665E-10 ], [ 8.7908523238589E-9 ] ]
    Error Statistics: {meanExponent=-8.378946343409662, negative=0, min=8.7908523238589E-9, max=8.7908523238589E-9, mean=5.561531200848346E-9, count=4.0, positive=4, stdDev=3.1410375605942524E-9, zeros=0}
    Finite-Difference Derivative Accuracy:
    absoluteTol: 2.0882e-08 +- 3.3264e-08 [9.8219e-10 - 1.0749e-07] (8#)
    relativeTol: 1.4624e-08 +- 1.5833e-08 [1.1358e-09 - 5.3755e-08] (8#)
    
```

Returns: 

```
    ToleranceStatistics{absoluteTol=2.0882e-08 +- 3.3264e-08 [9.8219e-10 - 1.0749e-07] (8#), relativeTol=1.4624e-08 +- 1.5833e-08 [1.1358e-09 - 5.3755e-08] (8#)}
```



### Performance
Code from [LayerTestBase.java:149](../../../../../../../src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L149) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2043 +- 0.4199 [0.1282 - 4.3573]
    Learning performance: 0.0031 +- 0.0026 [0.0000 - 0.0228]
    
```

