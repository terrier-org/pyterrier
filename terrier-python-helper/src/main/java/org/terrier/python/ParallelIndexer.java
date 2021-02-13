package org.terrier.python;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.concurrent.Callable;
import java.util.Arrays;
import java.lang.reflect.Constructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.terrier.indexing.Collection;
import org.terrier.indexing.CollectionFactory;
import org.terrier.structures.Index;
import org.terrier.structures.IndexOnDisk;
import org.terrier.structures.IndexUtil;
import org.terrier.structures.indexing.Indexer;
import org.terrier.structures.merging.BlockStructureMerger;
import org.terrier.structures.merging.StructureMerger;
import org.terrier.utility.ApplicationSetup;
import org.terrier.utility.TagSet;


/** Indexes sourceCollections in parallel to outputPath, using the provided indexerClass and mergerClass.
    It uses one thread per collection.
    Implementation based off org.terrier.applications.ThreadedBatchIndexing. */
public class ParallelIndexer {
    protected static Logger logger = LoggerFactory.getLogger(ParallelIndexer.class);

    public static void buildParallel(Collection[] sourceCollections, String output, Class indexerClass, Class mergerClass) {
        try {
            final Collection[] collections = sourceCollections;
            final String outputPath = output;
            final Constructor indexerConst = indexerClass.getConstructor(new Class[]{String.class, String.class}); // path, prefix
            final Constructor mergerConst = mergerClass.getConstructor(new Class[]{IndexOnDisk.class, IndexOnDisk.class, IndexOnDisk.class}); // a, b, out

            final long starttime = System.currentTimeMillis();
            final AtomicInteger indexCounter = new AtomicInteger();
            final AtomicInteger mergeCounter = new AtomicInteger();         
            
            final int threadCount = collections.length;

            IndexOnDisk.setIndexLoadingProfileAsRetrieval(false);
            final Function<Collection,String> indexer = new Function<Collection,String>()
            {
                @Override
                public String apply(Collection collection) {
                    String thisPrefix = "data_stream"+indexCounter.getAndIncrement();
                    try {
                        Indexer indexer = (Indexer) indexerConst.newInstance(outputPath, thisPrefix);
                        indexer.index(new Collection[] {collection});
                    } catch (Throwable ex) {
                        // TODO: better error handling
                        throw new RuntimeException(ex);
                    }
                    return thisPrefix;
                }   
            };
            final BinaryOperator<String> merger = new BinaryOperator<String>()
            {
                @Override
                public String apply(String t, String u) {
                    try {
                        if (t == null && u == null)
                            return null;
                        if (t == null)
                            return u;
                        if (u == null)
                            return t;
                        Index.setIndexLoadingProfileAsRetrieval(false);
                        IndexOnDisk src1 = IndexOnDisk.createIndex(outputPath, t);
                        IndexOnDisk src2 = IndexOnDisk.createIndex(outputPath, u);
                        if (src1 == null && src2 == null)
                            return null;
                        if (src1 == null)
                            return u;
                        if (src2 == null)
                            return t;
                        int doc1 = src1.getCollectionStatistics().getNumberOfDocuments();
                        int doc2 = src2.getCollectionStatistics().getNumberOfDocuments();
                        if (doc1 > 0 && doc2 == 0)
                        {
                            IndexUtil.deleteIndex(outputPath, u);
                            return t;
                        } else if (doc1 == 0 && doc2 > 0 ) {
                            IndexUtil.deleteIndex(outputPath, t);
                            return u;
                        } else if (doc1 == 0 && doc2 == 0) {
                            IndexUtil.deleteIndex(outputPath, t);
                            IndexUtil.deleteIndex(outputPath, u);
                            return null;
                        }
                        
                        String thisPrefix = "data_merge"+mergeCounter.getAndIncrement();
                        IndexOnDisk newIndex = IndexOnDisk.createNewIndex(outputPath, thisPrefix);
                        
                        StructureMerger indexMerger = (StructureMerger)mergerConst.newInstance(src1, src2, newIndex);
                        indexMerger.mergeStructures();
                        
                        src1.close();
                        src2.close();
                        newIndex.close();
                        //TODO: could index deletion occur in parallel
                        IndexUtil.deleteIndex(outputPath, t);
                        IndexUtil.deleteIndex(outputPath, u);
                        return thisPrefix;
                    } catch (Throwable e) {
                        // TODO: better error handling
                        throw new RuntimeException(e);
                    }
                }
            };
            
            Callable<String> callable = new Callable<String>() {
                @Override
                public String call() {
                    return Arrays.asList(collections).parallelStream().map(indexer).reduce(merger).get();
                }
            };
            ForkJoinPool forkPool = new ForkJoinPool(threadCount);
            String tmpPrefix = forkPool.submit(callable).get();
            if (tmpPrefix == null)
            {
                return;
            }
            IndexUtil.renameIndex(outputPath, tmpPrefix, outputPath, "data");
        } catch (Throwable e) {
            // TODO: do better than catching everything
            logger.error("Problem occurred during parallel indexing", e);
        }
    }
}