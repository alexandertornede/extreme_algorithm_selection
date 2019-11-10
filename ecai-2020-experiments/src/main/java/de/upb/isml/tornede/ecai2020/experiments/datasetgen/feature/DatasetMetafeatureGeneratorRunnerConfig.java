package de.upb.isml.tornede.ecai2020.experiments.datasetgen.feature;

import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.basic.IDatabaseConfig;

@Sources({ "file:./conf/dataset_metafeature_generator_runner.properties" })
public interface DatasetMetafeatureGeneratorRunnerConfig extends IDatabaseConfig {

}
