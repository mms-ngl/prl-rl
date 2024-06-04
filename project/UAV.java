/**
 * A single UAV running over the simulation. 
 * This class implements the class Steppable, the latter requires the implementation 
 * of one crucial method: step(SimState).
 * Please refer to Mason documentation for further details about the step method and how the simulation
 * loop is working.
 * 
 * @author dario albani
 * @mail albani@dis.uniroma1.it
 * @thanks Sean Luke
 */
package sim.app.firecontrol;

import java.util.LinkedHashSet;
import java.util.ArrayList; 
import java.util.LinkedList;
import java.util.Set;
import java.util.Map;

import java.util.Random;

import sim.engine.SimState;
import sim.engine.Steppable;
import sim.util.Double3D;
import sim.util.Int3D;
import sim.util.Int2D;

public class UAV implements Steppable{
	private static final long serialVersionUID = 1L;

	// Agent's variable
	public int id; //unique ID
	public double x; //x position in the world
	public double y; //y position in the world
	public double z; //z position in the world
	public Double3D target; //UAV target
	public AgentAction action; //last action executed by the UAV
	// Agent's local knowledge 
	public Set<WorldCell> knownCells; 
	public LinkedList<Integer> visitedStates; 
	public Task myTask;
	
	// Agent's settings - static because they have to be the same for all the 
	// UAV in the simulation. If you change it once, you change it for all the UAV.
	public static double linearvelocity = 0.02;

	//used to count the steps needed to extinguish a fire in a location
	public static int stepToExtinguish = 10;
	//used to remember when first started to extinguish at current location
	private int startedToExtinguishAt = -1;

    public static int statesCount;
    public static double epsilon = 0.1;
    public static int N = 50; //number of experiences to store in M

    public int crtState;

	public UAV(int id, Double3D myPosition){
		//set agent's id
		this.id = id;
		//set agent's position
		this.x = myPosition.x;
		this.y = myPosition.y;
		this.z = myPosition.z;
		//at the beginning agents have no action
		this.action = null;
		//at the beginning agents have no known cells 
		this.knownCells = new LinkedHashSet<>();
		this.visitedStates = new LinkedList<>();

		statesCount = Ignite.statesCount;

	}

	// DO NOT REMOVE
	// Getters and setters are used to display information in the inspectors
	public int getId(){
		return this.id;
	}

	public void setId(int id){
		this.id = id;
	}

	public double getX(){
		return this.x;
	}

	public void setX(double x){
		this.x = x;
	}

	public double getY(){
		return this.y;
	}

	public void setY(double y){
		this.y = y;
	}

	public double getZ(){
		return this.z;
	}

	public void setZ(double z){
		this.z = z;
	}

	/**
	 *  Do one step.
	 *  Core of the simulation.   
	 */
	public void step(SimState state){
		Ignite ignite = (Ignite)state;

		AgentAction a = nextAction(ignite);

		//select the next action for the agent
		switch(a){
		case SELECT_TASK:

			selectTask(ignite); 

			this.action = a; 
			break;
		case SELECT_CELL:

			selectCell(ignite); 
			
		case MOVE:
			move(state);
			break;

		case EXTINGUISH:
			//if true set the cell to be normal and foamed
			if(extinguish(ignite)){
				//retrieve discrete location of this
				Int3D dLoc = ignite.air.discretize(new Double3D(this.x, this.y, this.z));
				//extinguish the fire
				((WorldCell)ignite.forest.field[dLoc.x][dLoc.y]).extinguish(ignite);
				this.target=null;
			}

			this.action = a;
			break;

		default:	
			break;
		}
	}

	private AgentAction nextAction(Ignite ignite){
		if(this.myTask == null){
			return AgentAction.SELECT_TASK;
		}
		//else, if I have a task but I do not have target I need to take one
		else if(this.target == null){
			return AgentAction.SELECT_CELL;
		}
		//else, if I have a target and task I need to move toward the target
		//check if I am over the target and in that case execute the right action;
		//if not, continue to move toward the target
		else if(this.target.equals(ignite.air.discretize(new Double3D(x, y, z)))){
			//if on fire then extinguish, otherwise move on
			WorldCell cell = (WorldCell)ignite.forest.field[(int) x][(int) y];

			//store the knowledge for efficient selection
			this.knownCells.add(cell);

			if(cell.type.equals(CellType.FIRE))
				return AgentAction.EXTINGUISH;
			else
				return AgentAction.SELECT_CELL;

		} else{
			return AgentAction.MOVE;
		}		
	}

	private void selectTask(Ignite ignite){
		this.myTask = ignite.task;
		this.target = new Double3D(this.myTask.centroid.x, this.myTask.centroid.y, z);
		
		int state = this.myTask.centroid.y * ignite.width + this.myTask.centroid.x;
       	this.crtState = state;
       	this.visitedStates.add(this.crtState);

		ignite.reward = ignite.reward + this.getR(ignite, crtState);
	}

	/**
	 * Take the centroid of the fire and its expected radius and select the next 
	 * cell that requires closer inspection or/and foam. 
	 */
	private void selectCell(Ignite ignite) {

        Random rand = new Random();

        int nextState;
        if(rand.nextDouble() > this.epsilon) {
        
			nextState = this.eGreedy(ignite, crtState);
        } else {

        	int[] actionsFromCurrentState = ignite.possibleActionsFromState(ignite, crtState);

       		int index = rand.nextInt(actionsFromCurrentState.length);
        	nextState = actionsFromCurrentState[index];
        }

        double reward = getR(ignite, nextState); 
        Buffer b = new Buffer(crtState, nextState, reward, nextState);

        if(ignite.M.size() < N){
       		ignite.M.add(b);
        } else {
        	ignite.M.removeFirst(); 
       		ignite.M.add(b);
        }

        crtState = nextState;

		ignite.reward = ignite.reward + this.getR(ignite, crtState);
       	this.visitedStates.add(this.crtState); //store next state in visited states list
            
        int x = crtState % ignite.width;
        int y = crtState / ignite.width;

		this.target = new Double3D(x, y, z);
	}

    int eGreedy(Ignite ignite, int state) {
    	/* get all possible actions from current state, then define actions
    	   which lead to not visited states. 
    	   If not visited set is empty, use all possible actions
    	*/
        int[] actionsFromState = ignite.possibleActionsFromState(ignite, state);
        ArrayList<Integer> temp = new ArrayList<>();
		for(int action : actionsFromState) {

            if(!this.visitedStates.contains(action)){
            	temp.add(action);
            }
        }

        int[] notVisitedStates = temp.stream().mapToInt(i -> i).toArray();

        int[] actions = notVisitedStates;

        int act = 0;
        if(notVisitedStates.length == 0){
        	actions = actionsFromState;
        } 

        double maxValue = Double.NEGATIVE_INFINITY;
        int counter = 0;
        for (int action : actions) {
            double value = Ignite.Q[state][action];
            if (value > maxValue){
                maxValue = value;
                act = action;
            }
        }

        return act;
    }

    //get reward
	double getR(Ignite ignite, int nextState){
		WorldCell nextCell = (WorldCell)ignite.forest.field[nextState % ignite.width][(int)nextState / ignite.width];

		double r = 0.0;
		if(nextCell.type.equals(CellType.NORMAL)){
			r = 0.0;
		} else if(nextCell.type.equals(CellType.EXTINGUISHED)){
			r = -0.25;
		} else if(nextCell.type.equals(CellType.FIRE)){
			r = 1.0;
		} else if(nextCell.type.equals(CellType.BURNED)){
			r = -0.5;
		}

		return r;
	}

	/**
	 * Move the agent toward the target position
	 * The agent moves at a fixed given velocity
	 * @see this.linearvelocity
	 */
	public void move(SimState state){
		Ignite ignite = (Ignite) state;

		// retrieve the location of this 
		Double3D location = ignite.air.getObjectLocationAsDouble3D(this);
		double myx = location.x;
		double myy = location.y;
		double myz = location.z;

		// compute the distance w.r.t. the target
		// the z axis is only used when entering or leaving an area
		double xdistance = this.target.x - myx;
		double ydistance = this.target.y - myy;

		if(xdistance < 0)
			myx -= Math.min(Math.abs(xdistance), linearvelocity);
		else
			myx += Math.min(xdistance, linearvelocity);

		if(ydistance < 0){ 
			myy -= Math.min(Math.abs(ydistance), linearvelocity); 
		}
		else{	
			myy += Math.min(ydistance, linearvelocity); 
		}

		// update position in the simulation
		ignite.air.setObjectLocation(this, new Double3D(myx, myy, myz));
		// update position local position
		this.x = myx;
		this.y = myy;
		this.z = myz;
	}

	/**
	 * Start to extinguish the fire at current location.
	 * @return true if enough time has passed and the fire is gone, false otherwise
	 * @see this.stepToExtinguish
	 * @see this.startedToExtinguishAt
	 */
	private boolean extinguish(Ignite ignite){
		if(startedToExtinguishAt==-1){
			this.startedToExtinguishAt = (int) ignite.schedule.getSteps();
		}
		//enough time has passed, the fire is gone
		if(ignite.schedule.getSteps() - startedToExtinguishAt == stepToExtinguish){
			startedToExtinguishAt = -1;
			return true;
		}		
		return false;
	}
	
	@Override
	public boolean equals(Object obj){
		UAV uav = (UAV) obj;
		return uav.id == this.id;
	}
	
	@Override
	public String toString(){ 
		return id+"UAV-"+x+","+y+","+z+"-"+action;
	} 	
}


