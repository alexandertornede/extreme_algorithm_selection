package ai.libs.jaicore.basic.kvstore;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;

import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection.EGroupMethod;

public class SQLResultsTableGenerator {
	private static final String DB_HOST = "";
	private static final String DB_USER = "";
	private static final String DB_PWD = "";
	private static final String DB_BASE = "";

	private static final List<String> INVERT_METRICS = Arrays.asList("0_kendalls_tau", "1_NDCG@3", "2_NDCG@5", "3_NDCG@10");

	private static Map<String, String> replacements;

	private static final String setting = "metric";
	private static final String sampleID = "approach";
	private static final String bestOutput = "best";
	private static final String metricResult = "metric_result";
	private static final String metricResultList = "metric_result_list";

	private static void initTextReplacements() {
		replacements = new HashMap<>();
		replacements.put("per_algorithm_regression_dyad_ranking_imitating_2_25", "01_PAReg");
		replacements.put("per_algorithm_regression_dyad_ranking_imitating_2_50", "02_PAReg");
		replacements.put("per_algorithm_regression_dyad_ranking_imitating_2_125", "03_PAReg");

		replacements.put("alors_cofirank_NDCG_dyad_ranking_imitating_2_25", "04_Alors (NDCG)");
		replacements.put("alors_cofirank_NDCG_dyad_ranking_imitating_2_50", "05_Alors (NDCG)");
		replacements.put("alors_cofirank_NDCG_dyad_ranking_imitating_2_125", "06_Alors (NDCG)");

		replacements.put("alors_cofirank_REGRESSION_dyad_ranking_imitating_2_25", "07_Alors (REGR)");
		replacements.put("alors_cofirank_REGRESSION_dyad_ranking_imitating_2_50", "08_Alors (REGR)");
		replacements.put("alors_cofirank_REGRESSION_dyad_ranking_imitating_2_125", "09_Alors (REGR)");

		replacements.put("real_dyad_ranker_random_2_25", "10_DR$_{2,25}$");
		replacements.put("real_dyad_ranker_random_2_50", "11_DR$_{2,50}$");
		replacements.put("real_dyad_ranker_random_2_125", "12_DR$_{2,125}$");

		replacements.put("2xfeature_regression_dyad_ranking_imitating_2_25", "13_DFReg");
		replacements.put("2xfeature_regression_dyad_ranking_imitating_2_50", "14_DFReg");
		replacements.put("2xfeature_regression_dyad_ranking_imitating_2_125", "15_DFReg");

		replacements.put("random", "16_RandomRank");

		replacements.put("average_performance_dyad_ranking_imitating_2_25", "17_AvgPerformance");
		replacements.put("average_performance_dyad_ranking_imitating_2_50", "18_AvgPerformance");
		replacements.put("average_performance_dyad_ranking_imitating_2_125", "19_AvgPerformance");

		replacements.put("1_nn_euclideandistance_dyad_ranking_imitating_2_25", "20_1-NN LR");
		replacements.put("1_nn_euclideandistance_dyad_ranking_imitating_2_50", "21_1-NN LR");
		replacements.put("1_nn_euclideandistance_dyad_ranking_imitating_2_125", "22_1-NN LR");

		replacements.put("2_nn_euclideandistance_dyad_ranking_imitating_2_25", "23_2-NN LR");
		replacements.put("2_nn_euclideandistance_dyad_ranking_imitating_2_50", "24_2-NN LR");
		replacements.put("2_nn_euclideandistance_dyad_ranking_imitating_2_125", "25_2-NN LR");

		replacements.put("kendalls_apache_ranks", "0_kendalls_tau");
		replacements.put("NDCG@3", "1_NDCG@3");
		replacements.put("NDCG@5", "2_NDCG@5");
		replacements.put("NDCG@10", "3_NDCG@10");
		replacements.put("best_top_1", "4_regret@1");
		replacements.put("best_top_3", "5_regret@3");
		replacements.put("best_top_5", "6_regret@5");
	}

	public static void main(final String[] args) throws SQLException, IOException {
		initTextReplacements();
		SQLAdapter adapter = new SQLAdapter(DB_HOST, DB_USER, DB_PWD, DB_BASE);

		KVStoreCollection col = KVStoreUtil.readFromMySQLTable(adapter, "IJCAI_Summary_DR", new HashMap<>());
		col.addAll(KVStoreUtil.readFromMySQLTable(adapter, "IJCAI_Summary_PAReg_DFReg", new HashMap<>()));
		col.addAll(KVStoreUtil.readFromMySQLTable(adapter, "IJCAI_Summary_Alors", new HashMap<>()));
		col.addAll(KVStoreUtil.readFromMySQLTable(adapter, "IJCAI _Summary_Baselines", new HashMap<>()));
		col.removeAny(new String[] { "avg_top_3", "avg_top_5" }, true);
		Map<String, String> remCondition = new HashMap<>();
		remCondition.put("approach", "per_algorithm_regression_random");
		col.removeAny(remCondition, true);
		Map<String, String> condition = new HashMap<>();
		condition.put(setting, "kendalls_apache");
		col.removeAny(condition, true);

		col.removeAny("bayesianAveraging");
		col.removeAny("average_rank");

		KVStoreCollection randCol = KVStoreUtil.readFromMySQLTable(adapter, "IJCAI_Summary_RandomRank", new HashMap<>());
		randCol.setCollectionID("randCol");
		KVStoreCollection randColCopy = new KVStoreCollection(randCol.toString());
		randColCopy.stream().forEach(x -> x.put("approach", x.getAsString("approach") + "_25"));
		col.addAll(randColCopy);
		randColCopy = new KVStoreCollection(randCol.toString());
		randColCopy.stream().forEach(x -> x.put("approach", x.getAsString("approach") + "_50"));
		col.addAll(randColCopy);
		randColCopy = new KVStoreCollection(randCol.toString());
		randColCopy.stream().forEach(x -> x.put("approach", x.getAsString("approach") + "_125"));
		col.addAll(randColCopy);

		for (IKVStore store : col) {
			if (store.getAsString("approach").contains("_")) {
				String[] split = store.getAsString("approach").split("\\_");
				if (split[split.length - 1].equals("bayesianAveraging")) {
					store.put("trainingExamples", split[split.length - 2]);
				} else {
					store.put("trainingExamples", split[split.length - 1]);
				}
			}
		}

		col.applyFilter(sampleID, new IKVFilter() {
			@Override
			public Object filter(final Object value) {
				if (replacements.containsKey(value)) {
					return replacements.get(value);
				}
				return value;
			}
		});
		col.applyFilter(setting, new IKVFilter() {
			@Override
			public Object filter(final Object value) {
				if (replacements.containsKey(value)) {
					return replacements.get(value);
				}
				return value;
			}
		});
		col.serializeTo(new File("data/experiment_data.kvstore"));
		System.exit(0);

		// invert some metrics in order to make all target functions to be minimized
		col.stream().filter(x -> INVERT_METRICS.contains(x.getAsString("metric"))).forEach(x -> x.put(metricResult, (-1) * x.getAsDouble(metricResult)));

		// group the kvstores by approaches and metrics
		Map<String, EGroupMethod> grouping = new HashMap<>();
		grouping.put(metricResult, EGroupMethod.AVG);
		grouping.put("n", EGroupMethod.MIN);
		grouping.put("trainingExamples", EGroupMethod.AVG);
		col = col.group(new String[] { sampleID, setting }, grouping);

		// ensure the kvstores to be sorted in order
		col.sort(new Comparator<IKVStore>() {
			@Override
			public int compare(final IKVStore o1, final IKVStore o2) {
				int compare = o1.getAsString(sampleID).compareTo(o2.getAsString(sampleID));
				if (compare != 0) {
					return compare;
				}
				return o1.getAsString(setting).toLowerCase().compareTo(o2.getAsString(setting).toLowerCase());
			}
		});

		Map<Integer, Set<String>> sampleMap = new HashMap<>();

		for (IKVStore store : col) {
			sampleMap.computeIfAbsent(store.getAsInt("GROUP_SIZE"), t -> new HashSet<>()).add(store.getAsString("approach"));
		}
		System.out.println(sampleMap);

		KVStoreCollectionPartition xPartition = new KVStoreCollectionPartition("trainingExamples", col);
		for (Entry<String, KVStoreCollection> entry : xPartition) {
			KVStoreCollection subCol = entry.getValue();
			// determine who's best
			KVStoreStatisticsUtil.best(subCol, setting, sampleID, metricResult);

			// carry out pair-wise wilcoxon signed rank tests comparing one-vs-rest with one being the best approach.
			TwoLayerKVStoreCollectionPartition partition = new TwoLayerKVStoreCollectionPartition(setting, sampleID, subCol);
			for (Entry<String, Map<String, KVStoreCollection>> partitionEntry : partition) {
				Optional<Entry<String, KVStoreCollection>> best = partitionEntry.getValue().entrySet().stream().filter(x -> x.getValue().get(0).getAsBoolean(bestOutput)).findFirst();
				if (best.isPresent()) {
					KVStoreCollection merged = new KVStoreCollection();
					partitionEntry.getValue().values().forEach(merged::addAll);
					try {
						KVStoreStatisticsUtil.wilcoxonSignedRankTest(merged, setting, sampleID, "dataset_split", metricResultList, best.get().getValue().get(0).getAsString("approach"), "wilcoxon");
					} catch (Exception e) {
						System.out.println(merged);
					}
				}
			}
		}

		// restore the inverted metrics
		col.stream().filter(x -> INVERT_METRICS.contains(x.getAsString("metric"))).forEach(x -> x.put(metricResult, (-1) * x.getAsDouble(metricResult)));

		// prepare entry for output
		col.stream().forEach(x -> {
			x.put(sampleID, "\\texttt{" + x.getAsString(sampleID) + "}");
			x.put(metricResult, ValueUtil.valueToString(x.getAsDouble(metricResult), 4));
			if (x.getAsBoolean("best")) {
				x.put(metricResult, "\\textbf{" + x.getAsString(metricResult) + "}");
			}

			switch (x.getAsString("wilcoxon")) {
			case "SUPERIOR":
				x.put(metricResult, x.getAsString(metricResult) + " $\\circ$");
				break;
			case "INFERIOR":
				x.put(metricResult, x.getAsString(metricResult) + " $\\bullet$");
				break;
			case "TIE":
				x.put(metricResult, x.getAsString(metricResult) + " $\\phantom{\\circ}$");
				break;
			default:
				break;
			}
		});

		// generate a latex table from the collection
		String latexTable = KVStoreUtil.kvStoreCollectionToLaTeXTable(col, sampleID, setting, metricResult);
		String lineSplit[] = latexTable.split("\n");
		System.out.println(lineSplit.length);
		for (int i = 0; i < lineSplit.length; i++) {
			if (i == 0) {
				System.out.println(lineSplit[i]);
				continue;
			}
			if (i == lineSplit.length - 1) {
				System.out.println(lineSplit[i]);
				continue;
			}
			if ((i - 1) % 3 == 0) {
				System.out.println(lineSplit[i].substring(0, lineSplit[i].length() - 2) + " ");
			} else if ((i - 1) % 3 == 1) {
				System.out.println(lineSplit[i].substring(lineSplit[i].indexOf("&"), lineSplit[i].length() - 2));
			} else if ((i - 1) % 3 == 2) {
				System.out.println(lineSplit[i].substring(lineSplit[i].indexOf("&")));
			} else {
			}

		}

		System.out.println(latexTable);

	}

}
