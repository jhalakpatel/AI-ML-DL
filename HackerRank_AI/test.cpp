/*
 Assumption : 
    1. videos with same videos - arranged in lexicographical order
    

 Approach :
    1. getRanking needs to faster - can implement a simple lookup for "videoID" and get the ranking
    2. videoViewed will update the views - will update the overall product ranking
    3. top 10 views - will simply iterate over the ordered list of views and return 10 views
*/

/*
    videoId - string to videoID
    pair<int,int> - (rank, viewo) pair
*/
#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<unordered_map>
#include<set>

using namespace std;
// global variables

unordered_map<string, pair<int,int>> videoIdToRankAndView;
map<int,set<string>> viewToVideos;      // same number of views, ranks are ordered by videoID - lexicographical order
int total_videos = 0;
void videoViewed(string videoId){
    cout << "videoViewed called" << endl;
    // video is viewed for the first time
    if (videoIdToRankAndView.find(videoId)==videoIdToRankAndView.end()) {
		total_videos++;
		// watching for the first time, add to view == 1
		if (viewToVideos.find(1)==viewToVideos.end()) {	
        	videoIdToRankAndView[videoId] = make_pair(total_videos-0,1);         // video will get zero rank and one view
        	// get list of videos with "1" view i.e. rank 0
       		 viewToVideos[1].emplace(videoId); 
		} else {
			cout << " already have video size : 1 " << endl;
			viewToVideos[1].emplace(videoId);
        	videoIdToRankAndView[videoId] = make_pair(total_videos-(viewToVideos[1].size()-1),1);         // video will get zero rank and one view
		}

        cout << "done adding a view to video " << videoId << " at rank : " << viewToVideos[1].size()-1 << " with views : " << 1 << endl;
        return;
    } else {
        cout << "videos already seen update : " << videoId << endl;
        pair<int,int> rankAndViews = videoIdToRankAndView[videoId];     // get rank and views for a video
        int views = rankAndViews.second;
        cout << "views : " << views << endl;
        // erase the video in previous mapping
        set<string> &videosWithGivenRank = viewToVideos[views];
        // erase the video in the mapping
        videosWithGivenRank.erase(videoId);
        
        // if the views mapping is empty - erase the mapping
        if (videosWithGivenRank.empty()) {
            viewToVideos.erase(views);
        }
        
        views++;    // increase the number of views for the given video

		cout << "increased views : " << views << endl;
        // store the new views
        viewToVideos[views].emplace(videoId);

		cout << "size of increased view : " <<viewToVideos[views].size() << endl;
       
        // iterate over all the views - not optimal solution, need to figure out way to get get rank optimally
        int r = 0;
        for (auto sitr=viewToVideos.begin(); sitr!=viewToVideos.end(); ++sitr){

			// print the views from the reverse itr
			cout << "views maximum to min : " << sitr->first << endl;

            if (sitr->first < views) {
				cout << "adding ranks " << sitr->second.size() << endl;
                r += sitr->second.size();
            } else {
				// found something with equal rank
				cout << "found with equal rank " << sitr->first << " view : " << views << endl;
                // get the rank among the current level with same number of views
                r += distance(sitr->second.find(videoId), sitr->second.begin());
                cout << "getting new rank : " << r << endl;
                break;
            }
        }
        // update the rank with new rank (directly update the reference)
		videoIdToRankAndView[videoId] = make_pair(total_videos-r, views);
        cout << "done adding a view to video " << videoId << " at rank : " << total_videos-r << " with views : " << views << endl;
        return;
    }
    return;
}

int getRanking(string videoId){
    cout << "get ranking called" << endl;
    // if the view is not viewed, its rank is "-1", not maxRank
    // check if the video has been viewed or not, return rank if the video has been viewed before
    if (videoIdToRankAndView.find(videoId)!=videoIdToRankAndView.end()) {
        cout << "rank for : " << videoId << " : " << videoIdToRankAndView[videoId].first << endl;
        return videoIdToRankAndView[videoId].first;
    }
    return -1;  // video does not have any rank
}

vector<string> getToVideos() {
   
    // return top 10 videos - atmost 10 videos
    // if less than 10 videos has been seen, return "N" number of videos, N<10
    vector<string> top10videos;
    
    // start videos from highest views to lowest views
    for (auto sitr=viewToVideos.rbegin(); sitr!=viewToVideos.rend(); ++sitr){
        if (top10videos.size()+sitr->second.size() < 10) {
             top10videos.insert(top10videos.end(),sitr->second.begin(), sitr->second.end());
        } else {
             for (auto itr=sitr->second.begin(); itr!=sitr->second.end(); ++itr) {
                top10videos.emplace_back(*itr);
                if (top10videos.size()==10) break;
             }
         }
    }
    
    return top10videos;     
}

int main() {
    videoViewed("video1");
    videoViewed("video2");
    videoViewed("video3");
    videoViewed("video1");
    videoViewed("video1");
    videoViewed("video2");
    videoViewed("video3");
	
    cout << getRanking("video1") << endl;
    cout << getRanking("video2") << endl;
    cout << getRanking("video3") << endl;
    return 0;
}
