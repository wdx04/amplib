#pragma once

#include "amp_find_best_transform.h"

namespace amp
{
	// dense & sparse grayscale template matching
	// sample code:
	// concurrency::accelerator_view acc_view = concurrency::accelerator().create_view();
	// cv::Mat templ = cv::imread("templ.png", cv::IMREAD_GRAYSCALE);
	// cv::Mat target = cv::imread("target.png", cv::IMREAD_GRAYSCALE);
	// amp::dense_template_matcher matcher(acc_view, templ, target.size(), 4);
	// cv::Matx23f best_affine_transform = matcher.match(target);

	// find target angle in very low precision
	inline double guess_target_angle(const cv::Mat& templ, const cv::Mat& target, double angle_prec = 7.5)
	{
		int angle_step = cvRound(180.0 / angle_prec);
		std::vector<double> scores(angle_step * 2 + 1);
		std::vector<cv::Point> score_poses(angle_step * 2 + 1);
		cv::Point2f target_center(target.cols / 2.0f, target.rows / 2.0f);
#ifdef __TBB_parallel_for_H
		tbb::parallel_for(-angle_step, angle_step + 1, [&](int i)
			{
				double angle = i * angle_prec;
				cv::Mat rot = cv::getRotationMatrix2D(target_center, angle, 1.0);
				cv::Mat warppedTarget, result;
				cv::warpAffine(target, warppedTarget, rot, target.size(), cv::INTER_LINEAR);
				cv::matchTemplate(warppedTarget, templ, result, cv::TM_CCORR_NORMED);
				cv::Point poMin, poMax;
				double dmin, dmax;
				cv::minMaxLoc(result, &dmin, &dmax, &poMin, &poMax);
				scores[i + angle_step] = dmax;
				score_poses[i + angle_step] = poMax;
			});
#else
		for (int i = -angle_step; i <= angle_step; i++)
		{
			double angle = i * angle_prec;
			cv::Mat rot = cv::getRotationMatrix2D(target_center, angle, 1.0);
			cv::Mat warppedTarget, result;
			cv::warpAffine(target, warppedTarget, rot, target.size(), cv::INTER_LINEAR);
			cv::matchTemplate(warppedTarget, templ, result, cv::TM_CCORR_NORMED);
			cv::Point poMin, poMax;
			double dmin, dmax;
			cv::minMaxLoc(result, &dmin, &dmax, &poMin, &poMax);
			scores[i + angle_step] = dmax;
			score_poses[i + angle_step] = poMax;
		}
#endif
		size_t best_score_index = std::max_element(scores.cbegin(), scores.cend()) - scores.cbegin();
		double best_angle = (int(best_score_index) - angle_step) * angle_prec;
		return best_angle;
	}

	// find initial target position in low precision
	cv::Matx23f get_best_transform_by_match_templ(const cv::Mat& templ, const cv::Mat& target, double min_angle, double max_angle)
	{
		double angle_prec = std::asin(1.0 / std::max(templ.rows, templ.cols)) * 180.0 / CV_PI;
		double mean_angle = (min_angle + max_angle) / 2.0;
		int min_angle_step = cvRound((mean_angle - min_angle) / angle_prec);
		int max_angle_step = cvRound((max_angle - mean_angle) / angle_prec);
		std::vector<double> scores(min_angle_step + max_angle_step + 1);
		std::vector<cv::Point> score_poses(min_angle_step + max_angle_step + 1);
		cv::Point2f target_center(target.cols / 2.0f, target.rows / 2.0f);
		cv::Point2f templ_center(templ.cols / 2.0f, templ.rows / 2.0f);
#ifdef __TBB_parallel_for_H
		tbb::parallel_for(-min_angle_step, max_angle_step + 1, [&](int i)
			{
				double angle = mean_angle + i * angle_prec;
				cv::Mat rot = cv::getRotationMatrix2D(target_center, angle, 1.0);
				cv::Mat warppedTarget, result;
				cv::warpAffine(target, warppedTarget, rot, target.size(), cv::INTER_LINEAR);
				cv::matchTemplate(warppedTarget, templ, result, cv::TM_CCORR_NORMED);
				cv::Point poMin, poMax;
				double dmin, dmax;
				cv::minMaxLoc(result, &dmin, &dmax, &poMin, &poMax);
				scores[i + min_angle_step] = dmax;
				score_poses[i + min_angle_step] = poMax;
			});
#else
		for (int i = -min_angle_step; i <= max_angle_step; i++)
		{
			double angle = mean_angle + i * angle_prec;
			cv::Mat rot = cv::getRotationMatrix2D(target_center, angle, 1.0);
			cv::Mat warppedTarget, result;
			cv::warpAffine(target, warppedTarget, rot, target.size(), cv::INTER_LINEAR);
			cv::matchTemplate(warppedTarget, templ, result, cv::TM_CCORR_NORMED);
			cv::Point poMin, poMax;
			double dmin, dmax;
			cv::minMaxLoc(result, &dmin, &dmax, &poMin, &poMax);
			scores[i + min_angle_step] = dmax;
			score_poses[i + min_angle_step] = poMax;
		}
#endif
		size_t best_score_index = std::max_element(scores.cbegin(), scores.cend()) - scores.cbegin();
		double best_angle = mean_angle + (int(best_score_index) - min_angle_step) * angle_prec;
		cv::Matx23f best_transform = cv::getRotationMatrix2D(target_center, best_angle, 1.0);
		cv::Matx23f best_transform_inversed;
		cv::invertAffineTransform(best_transform, best_transform_inversed);
		best_transform_inversed(0, 2) += float(score_poses[best_score_index].x * best_transform_inversed(0, 0) + score_poses[best_score_index].y * best_transform_inversed(0, 1));
		best_transform_inversed(1, 2) += float(score_poses[best_score_index].x * best_transform_inversed(1, 0) + score_poses[best_score_index].y * best_transform_inversed(1, 1));
		return best_transform_inversed;
	}

	class dense_template_matcher
	{
	public:
		dense_template_matcher(const accelerator_view& acc_view_, const cv::Mat& templ_, cv::Size target_size_, int levels_ = 4)
			: vctx(acc_view_, float(target_size_.width * target_size_.height) / 1000000.0f, size_t(levels_) * 2), target_size(target_size_), levels(levels_)
		{
			for (int level = 0; level < levels; level++)
			{
				int scale = 1 << level;
				if (level + 1 != levels)
				{
					cv::Mat scaled_templ;
					if (scale > 1)
					{
						cv::resize(templ_, scaled_templ, cv::Size(templ_.cols / scale, templ_.rows / scale), 0.0, 0.0, cv::INTER_LINEAR);
					}
					else
					{
						scaled_templ = templ_;
					}
					int templ_index = vctx.create_float2d_buf(scaled_templ.rows, scaled_templ.cols); // for template
					vctx.load_cv_mat(scaled_templ, vctx.float2d[templ_index]);
					vctx.create_float2d_buf(target_size_.height / scale, target_size_.width / scale); // for target
				}
				else
				{
					cv::resize(templ_, init_templ, cv::Size(templ_.cols / scale, templ_.rows / scale), 0.0, 0.0, cv::INTER_LINEAR);
				}
			}
		}

		cv::Matx23f match(const cv::Mat& target, float precision = 1.0f, double min_angle = -5.0, double max_angle = 5.0, bool use_init_guess = false)
		{
			if (target.size() != target_size)
			{
				throw std::runtime_error("target size mismatch in dense_template_matcher::match");
			}
			int level = levels - 1;
			int scale = 1 << level;
			cv::Mat init_target;
			cv::resize(target, init_target, cv::Size(target.cols / scale, target.rows / scale), 0.0, 0.0, cv::INTER_LINEAR);
			if (use_init_guess)
			{
				double init_angle = guess_target_angle(init_templ, init_target, 7.5);
				min_angle = init_angle - 5.0;
				max_angle = init_angle + 5.0;
			}
			cv::Matx23f current_transform = get_best_transform_by_match_templ(init_templ, init_target, min_angle, max_angle);
			do
			{
				level--;
				current_transform(0, 2) = current_transform(0, 2) * 2.0f;
				current_transform(1, 2) = current_transform(1, 2) * 2.0f;
				float level_precision = level == 0 ? precision : std::fmaxf(1.0f, precision);
				int templ_index = level * 2;
				int target_index = level * 2 + 1;
				int scale = 1 << level;
				cv::Mat scaled_target;
				cv::resize(target, scaled_target, cv::Size(vctx.float2d[target_index].extent[1], vctx.float2d[target_index].extent[0]), 0.0, 0.0, cv::INTER_LINEAR);
				vctx.load_cv_mat(scaled_target, vctx.float2d[target_index]);
				std::vector<cv::Matx23f> transforms;
				float templ_length = float(std::max(vctx.float2d[templ_index].extent[0], vctx.float2d[templ_index].extent[1]));
				float angle_precision = float(std::asinf(level_precision / templ_length) * 180.0 / CV_PI);
				int angle_step = std::max(3, cvRound(1.0f / level_precision) + 1);
				int xy_step = std::max(3, cvRound(1.0f / level_precision) + 1);
				float current_angle = float(std::atan2f(current_transform(0, 1), current_transform(0, 0)) * 180.0 / CV_PI);
				cv::Point2f templ_center(vctx.float2d[templ_index].extent[1] / 2.0f, vctx.float2d[templ_index].extent[0] / 2.0f);
				cv::Matx23f current_rot_matrix = cv::getRotationMatrix2D(templ_center, current_angle, 1.0);
				for (int angle = -angle_step; angle <= angle_step; angle++)
				{
					float angle0 = current_angle + angle * angle_precision;
					cv::Matx23f rot_matrix = cv::getRotationMatrix2D(templ_center, angle0, 1.0);
					float alpha = rot_matrix(0, 0);
					float beta = rot_matrix(0, 1);
					cv::Point2f init_loc(current_transform(0, 2) + rot_matrix(0, 2) - current_rot_matrix(0, 2), current_transform(1, 2) + rot_matrix(1, 2) - current_rot_matrix(1, 2));
					for (int x_delta = -xy_step; x_delta <= xy_step; x_delta++)
					{
						for (int y_delta = -xy_step; y_delta <= xy_step; y_delta++)
						{
							cv::Matx23f transform(alpha, beta, float(init_loc.x + x_delta * level_precision), -beta, alpha, float(init_loc.y + y_delta * level_precision));
							transforms.push_back(transform);
						}
					}
				}
				auto [score, best_transform] = amp::find_best_transform_inverse(vctx.acc_view, vctx.float2d[templ_index], vctx.float2d[target_index], transforms);
				current_transform = best_transform;
			} while (level > 0);
			return current_transform;
		}

	private:
		amp::vision_context vctx;
		cv::Mat init_templ;
		cv::Size target_size;
		int levels;
	};

	class sparse_template_matcher
	{
	public:
		sparse_template_matcher(const accelerator_view& acc_view_, const cv::Mat& templ_, cv::Size target_size_, double templ_sobel_thresh, int templ_dilation = 0, int levels_ = 4)
			: vctx(acc_view_, float(target_size_.width* target_size_.height) / 1000000.0f, size_t(levels_) * 2), target_size(target_size_), levels(levels_)
		{
			templ_size = templ_.size();
			for (int level = 0; level < levels; level++)
			{
				int scale = 1 << level;
				if (level + 1 != levels)
				{
					cv::Mat scaled_templ;
					if (scale > 1)
					{
						cv::resize(templ_, scaled_templ, cv::Size(templ_.cols / scale, templ_.rows / scale), 0.0, 0.0, cv::INTER_LINEAR);
					}
					else
					{
						scaled_templ = templ_;
					}

					cv::Mat grad = calculate_sobel(scaled_templ);
					cv::Mat grad_bin;
					cv::threshold(grad, grad_bin, templ_sobel_thresh, 255.0, cv::THRESH_BINARY);
					if (templ_dilation != 0)
					{
						cv::dilate(grad_bin, grad_bin, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)));
					}
					std::vector<cv::Point> edge_points;
					cv::findNonZero(grad_bin, edge_points);
					std::vector<cv::Vec3f> edge_values(edge_points.size());
					std::transform(edge_points.cbegin(), edge_points.cend(), edge_values.begin(), [&scaled_templ](cv::Point pt) {
						cv::Vec3f value;
						value[0] = float(pt.x);
						value[1] = float(pt.y);
						value[2] = float(scaled_templ.at<uchar>(pt));
						return value;
					});
					int templ_index = vctx.create_float2d_buf(edge_values.size(), 3); // for template
					concurrency::copy((const float*)&edge_values[0], vctx.float2d[templ_index]);
					vctx.create_float2d_buf(target_size_.height / scale, target_size_.width / scale); // for target
				}
				else
				{
					cv::resize(templ_, init_templ, cv::Size(templ_.cols / scale, templ_.rows / scale), 0.0, 0.0, cv::INTER_LINEAR);
				}
			}
		}

		cv::Mat calculate_sobel(const cv::Mat& src)
		{
			cv::Mat grad, grad_x, grad_y;
			cv::Mat scaled_grad_x, scaled_grad_y;
			cv::Sobel(src, grad_x, CV_32F, 1, 0, 5, 1, 0, cv::BORDER_DEFAULT);
			grad_x = cv::abs(grad_x);
			cv::Sobel(src, grad_y, CV_32F, 0, 1, 5, 1, 0, cv::BORDER_DEFAULT);
			grad_y = cv::abs(grad_y);
			cv::addWeighted(grad_x, 0.05, grad_y, 0.05, 0, grad);
			return grad;
		}

		cv::Matx23f match(const cv::Mat& target, float precision = 1.0f, double min_angle = -5.0, double max_angle = 5.0, bool use_init_guess = false)
		{
			if (target.size() != target_size)
			{
				throw std::runtime_error("target size mismatch in dense_template_matcher::match");
			}
			int level = levels - 1;
			int scale = 1 << level;
			cv::Mat init_target;
			cv::resize(target, init_target, cv::Size(target.cols / scale, target.rows / scale), 0.0, 0.0, cv::INTER_LINEAR);
			if (use_init_guess)
			{
				double init_angle = guess_target_angle(init_templ, init_target, 7.5);
				min_angle = init_angle - 5.0;
				max_angle = init_angle + 5.0;
			}
			cv::Matx23f current_transform = get_best_transform_by_match_templ(init_templ, init_target, min_angle, max_angle);
			do
			{
				level--;
				current_transform(0, 2) = current_transform(0, 2) * 2.0f;
				current_transform(1, 2) = current_transform(1, 2) * 2.0f;
				float level_precision = level == 0 ? precision : std::fmaxf(1.0f, precision);
				int templ_index = level * 2;
				int target_index = level * 2 + 1;
				int scale = 1 << level;
				cv::Size current_templ_size = templ_size / scale;
				cv::Mat scaled_target;
				cv::resize(target, scaled_target, cv::Size(vctx.float2d[target_index].extent[1], vctx.float2d[target_index].extent[0]), 0.0, 0.0, cv::INTER_LINEAR);
				vctx.load_cv_mat(scaled_target, vctx.float2d[target_index]);
				std::vector<cv::Matx23f> transforms;
				float templ_length = float(std::max(current_templ_size.width, current_templ_size.height));
				float angle_precision = float(std::asinf(level_precision / templ_length) * 180.0 / CV_PI);
				int angle_step = std::max(3, cvRound(1.0f / level_precision) + 1);
				int xy_step = std::max(3, cvRound(1.0f / level_precision) + 1);
				float current_angle = float(std::atan2f(current_transform(0, 1), current_transform(0, 0)) * 180.0 / CV_PI);
				cv::Point2f templ_center(float(current_templ_size.width) / 2.0f, float(current_templ_size.height) / 2.0f);
				cv::Matx23f current_rot_matrix = cv::getRotationMatrix2D(templ_center, current_angle, 1.0);
				for (int angle = -angle_step; angle <= angle_step; angle++)
				{
					float angle0 = current_angle + angle * angle_precision;
					cv::Matx23f rot_matrix = cv::getRotationMatrix2D(templ_center, angle0, 1.0);
					float alpha = rot_matrix(0, 0);
					float beta = rot_matrix(0, 1);
					cv::Point2f init_loc(current_transform(0, 2) + rot_matrix(0, 2) - current_rot_matrix(0, 2), current_transform(1, 2) + rot_matrix(1, 2) - current_rot_matrix(1, 2));
					for (int x_delta = -xy_step; x_delta <= xy_step; x_delta++)
					{
						for (int y_delta = -xy_step; y_delta <= xy_step; y_delta++)
						{
							cv::Matx23f transform(alpha, beta, float(init_loc.x + x_delta * level_precision), -beta, alpha, float(init_loc.y + y_delta * level_precision));
							transforms.push_back(transform);
						}
					}
				}
				auto [score, best_transform] = amp::find_best_transform_inverse_sparse(vctx.acc_view, vctx.float2d[templ_index], vctx.float2d[target_index], transforms);
				current_transform = best_transform;
			} while (level > 0);
			return current_transform;
		}

	private:
		amp::vision_context vctx;
		cv::Mat init_templ;
		cv::Size templ_size;
		cv::Size target_size;
		int levels;
	}; 
}
