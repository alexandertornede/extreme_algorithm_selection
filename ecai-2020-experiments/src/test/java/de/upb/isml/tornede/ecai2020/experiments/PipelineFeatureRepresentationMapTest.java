package de.upb.isml.tornede.ecai2020.experiments;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.SQLException;

public class PipelineFeatureRepresentationMapTest {

	public static void main(String[] args) throws SQLException, IOException {

		String fileContentNoctua = "#!/bin/bash\n" + //
				"#SBATCH -N 1\n" + //
				"#SBATCH -J ecai2020_ranking\n" + //
				"#SBATCH -A hpc-prf-isys\n" + //
				"#SBATCH -p batch\n" + //
				"#SBATCH -t 12:00:00\n" + //
				"#SBATCH --mail-type all\n" + //
				"#SBATCH --mail-user ahetzer@mail.upb.de\n" + //
				"\n" + "#run your application here\n" + //
				"export OMP_NUM_THREADS=2\n" + //
				"java -Xmx190G -XX:ParallelGCThreads=2 -jar ecai-2020-experiments-all.jar $P$";

		for (int i = 0; i < 10; i++) {
			String adaptedFileContent = fileContentNoctua.replace("$P$", "" + i);
			Files.writeString(Paths.get("conf/run-experiment_" + i + ".sh"), adaptedFileContent);
			System.out.println("ccsalloc run-experiment_" + i + ".sh");
		}

		for (int i = 0; i < 10; i++) {
			System.out.println("sbatch run-experiment_" + i + ".sh");
		}
	}

}
