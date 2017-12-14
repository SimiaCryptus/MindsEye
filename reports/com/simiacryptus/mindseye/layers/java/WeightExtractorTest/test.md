# WeightExtractor
## WeightExtractorTest
### Json Serialization
Code from [StandardLayerTests.java:68](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L68) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.java.WeightExtractor",
      "id": "dd42ee8b-0bc0-4d59-8401-3a59bcfdacbb",
      "isFrozen": false,
      "name": "WeightExtractor/dd42ee8b-0bc0-4d59-8401-3a59bcfdacbb",
      "innerId": "6f3582a9-d5b2-446a-a1b1-43d1f240b02b",
      "index": 0
    }
```



### Example Input/Output Pair
Code from [StandardLayerTests.java:152](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L152) executed in 0.00 seconds: 
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
    [[  ]]
    --------------------
    Output: 
    [ -0.6572861405986538, 0.1640098995032478, -0.29574910960558043, 0.2460078376634364, 0.5740674565273136, 0.5002539364494496, 0.1947868942782986, -0.76381992262552, 0.7687883767713144 ]
    --------------------
    Derivative: 
    [  ]
```



### Performance
Code from [StandardLayerTests.java:119](../../../../../../../src/main/java/com/simiacryptus/mindseye/test/StandardLayerTests.java#L119) executed in 0.00 seconds: 
```java
    getPerformanceTester().test(layer, permPrototype);
```
Logging: 
```
    100 batches
    Input Dimensions:
    	[0]
    Performance:
    	Evaluation performance: 0.000021s +- 0.000008s [0.000014s - 0.000036s]
    	Learning performance: 0.000006s +- 0.000003s [0.000003s - 0.000009s]
    
```

