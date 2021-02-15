package org.terrier.python;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.concurrent.Callable;
import java.util.Arrays;
import java.lang.reflect.Constructor;
import java.lang.ReflectiveOperationException;
import java.io.IOException;
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
  * It uses one thread per collection.
  * Implementation based off org.terrier.applications.ThreadedBatchIndexing. */
public class ParallelIndexer {
    protected static Logger logger = LoggerFactory.getLogger(ParallelIndexer.class);

    public static void buildParallel(Collection[] sourceCollections, String output, Class indexerClass, Class mergerClass) {
        final Collection[] collections = sourceCollections;
        final String outputPath = output;
        final Constructor indexerConst;
        final Constructor mergerConst;
        try {
            indexerConst = indexerClass.getConstructor(new Class[]{String.class, String.class}); // path, prefix
            mergerConst = mergerClass.getConstructor(new Class[]{IndexOnDisk.class, IndexOnDisk.class, IndexOnDisk.class}); // a, b, out
        } catch (ReflectiveOperationException ex) {
            throw new RuntimeException(ex); // Thrown if mergerClass is not for an StructureMerger class
        }

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
                Indexer indexer;
                try {
                    indexer = (Indexer) indexerConst.newInstance(outputPath, thisPrefix);
                } catch (ReflectiveOperationException ex) {
                    throw new RuntimeException(ex); // Thrown if indexerClass is not for an Indexer class
                }
                indexer.index(new Collection[] {collection});
                return thisPrefix;
            }   
        };
        final BinaryOperator<String> merger = new BinaryOperator<String>()
        {
            @Override
            public String apply(String t, String u) {
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
                try {
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
                } catch (IOException ex) {
                    throw new RuntimeException(ex); // problem deleting index
                }
                
                String thisPrefix = "data_merge"+mergeCounter.getAndIncrement();
                IndexOnDisk newIndex = IndexOnDisk.createNewIndex(outputPath, thisPrefix);
                
                StructureMerger indexMerger;
                try {
                    indexMerger = (StructureMerger)mergerConst.newInstance(src1, src2, newIndex);
                } catch (ReflectiveOperationException ex) {
                    throw new RuntimeException(ex); // Thrown if mergerClass is not for an StructureMerger class
                }
                indexMerger.mergeStructures();
                
                try {
                    src1.close();
                    src2.close();
                    newIndex.close();
                    //TODO: could index deletion occur in parallel
                    IndexUtil.deleteIndex(outputPath, t);
                    IndexUtil.deleteIndex(outputPath, u);
                } catch (IOException ex) {
                    throw new RuntimeException(ex); // Thrown if problrm saving or deleting index
                }
                return thisPrefix;
            }
        };
        
        Callable<String> callable = new Callable<String>() {
            @Override
            public String call() {
                return Arrays.asList(collections).parallelStream().map(indexer).reduce(merger).get();
            }
        };
        ForkJoinPool forkPool = new ForkJoinPool(threadCount);
        String tmpPrefix;
        try {
            tmpPrefix = forkPool.submit(callable).get();
        } catch (java.lang.InterruptedException ex) {
            throw new RuntimeException(ex); // Problem running thread
        } catch (java.util.concurrent.ExecutionException ex) {
            throw new RuntimeException(ex); // Problem running thread
        }
        if (tmpPrefix == null)
        {
            return;
        }
        try {
            IndexUtil.renameIndex(outputPath, tmpPrefix, outputPath, "data");
        } catch (IOException ex) {
            throw new RuntimeException(ex); // Problem renaming index
        }
    }
}