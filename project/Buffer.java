
package sim.app.firecontrol;
import java.util.LinkedList;


public class Buffer {
	public int crtState;
	public int action;
	public double reward;
	public int nextState;
	public Buffer(int crtState, int action, double reward, int nextState) {
		this.crtState = crtState;
		this.action = action;
		this.reward = reward;
		this.nextState = nextState;
	}
}