package ai.libs.jaicore.basic.kvstore;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection.EGroupMethod;

public class ResultsTableGenerator {

	private static final List<String> INVERT_METRICS = Arrays.asList("0_kendalls_apache_ranks", "1_NDCG@3", "2_NDCG@5", "3_NDCG@10");

	private static final String setting = "metric";
	private static final String sampleID = "approach";
	private static final String bestOutput = "best";
	private static final String metricResult = "metric_result";
	private static final String metricResultList = "metric_result_list";

	public static void main(final String[] args) throws SQLException, IOException {
		KVStoreCollection col = new KVStoreCollection(FileUtil.readFileAsString(new File("data/experiment_data.kvstore")));
		// invert some metrics in order to make all target functions to be minimized
		col.stream().filter(x -> INVERT_METRICS.contains(x.getAsString("metric"))).forEach(x -> x.put(metricResult, (-1) * x.getAsDouble(metricResult)));

		// group the kvstores by approaches and metrics
		Map<String, EGroupMethod> grouping = new HashMap<>();
		grouping.put(metricResult, EGroupMethod.AVG);
		grouping.put("n", EGroupMethod.MIN);
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

		// determine who's best
		KVStoreStatisticsUtil.best(col, setting, sampleID, metricResult);

		// carry out pair-wise wilcoxon signed rank tests comparing one-vs-rest with one being the best approach.
		TwoLayerKVStoreCollectionPartition partition = new TwoLayerKVStoreCollectionPartition(setting, sampleID, col);
		for (Entry<String, Map<String, KVStoreCollection>> partitionEntry : partition) {
			Optional<Entry<String, KVStoreCollection>> best = partitionEntry.getValue().entrySet().stream().filter(x -> x.getValue().get(0).getAsBoolean(bestOutput)).findFirst();
			if (best.isPresent()) {
				KVStoreCollection merged = new KVStoreCollection();
				partitionEntry.getValue().values().forEach(merged::addAll);
				KVStoreStatisticsUtil.wilcoxonSignedRankTest(merged, setting, sampleID, "dataset_split", metricResultList, best.get().getValue().get(0).getAsString("approach"), "wilcoxon");
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
		System.out.println(latexTable);
	}
}
