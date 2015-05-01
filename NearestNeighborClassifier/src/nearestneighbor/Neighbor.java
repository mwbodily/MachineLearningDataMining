/*
 * This is a basic class to contain the index and distance of a node in the
 * kNN classifier. Note that the index is the node's position in the list
 * of training instances.
 */
package nearestneighbor;

/**
 * @author Mackenzie Bodily
 */
public class Neighbor {

        public int index;
        public double distance;
        
        /********************************************************************
         * The default constructor. Forms a default Neighbor class with a 
         * negative index and a distance of infinity.
         *********************************************************************/
        public Neighbor()
        {
            index = -1;
            distance = Double.POSITIVE_INFINITY;        
        }
        
        /********************************************************************
         * Constructor that allows the user to set the index and distance
         * of one node from the training set to the instance contained
         * in the data set. 
         * 
         * @param pIndex - the neighbor's position in the dataSet instances
         * @param pDistance - the distance of the neighbor from the tested
         *                    instance
         ********************************************************************/
        public Neighbor(int pIndex, double pDistance)
        {
            index = pIndex;
            distance = pDistance;      
        }
}
